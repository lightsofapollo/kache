//! Whether the build that asks for a permit runs in the foreground.
//!
//! The answer describes the calling build, not the process that waits: a
//! request the daemon executes carries its client's priority in
//! [`InvocationPriorityScope`], and every other request reads its own
//! process. A child inherits its parent's priority, so a compiler wrapper
//! reports the priority of the Cargo, make or IDE that started it.
//!
//! - macOS: Darwin background (`taskpolicy -b`, `setpriority(PRIO_DARWIN_*)`)
//!   or a positive nice value.
//! - Linux: a positive nice value, or the `SCHED_BATCH` / `SCHED_IDLE`
//!   policy.
//! - Windows: the idle or below-normal priority class.
//! - Elsewhere, and whenever the process cannot be read: foreground.

/// Scheduling class of the build behind a permit request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Priority {
    #[default]
    Foreground,
    Background,
}

impl Priority {
    fn from_background(background: bool) -> Self {
        if background {
            Self::Background
        } else {
            Self::Foreground
        }
    }

    /// The priority of this process.
    pub(crate) fn of_this_process() -> Self {
        #[cfg(unix)]
        {
            Self::of_process(std::process::id())
        }
        #[cfg(windows)]
        {
            Self::from_background(this_process_is_background())
        }
        #[cfg(not(any(unix, windows)))]
        {
            Self::Foreground
        }
    }

    /// The priority of the process `pid`. The caller must know `pid` still
    /// names the process it means, such as the authenticated peer of an open
    /// connection.
    #[cfg(unix)]
    pub(crate) fn of_process(pid: u32) -> Self {
        Self::from_background(process_is_background(pid))
    }

    /// The priority a Windows priority class stands for.
    #[cfg(any(test, windows))]
    pub(crate) fn of_windows_class(class: u32) -> Self {
        Self::from_background(windows_background(class))
    }
}

/// The process that sent a request the daemon executes, and its priority
/// when the request arrived. The caller must know `pid` names that process
/// for as long as the request runs, such as the authenticated peer of the
/// open connection the request came on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Client {
    pub(crate) priority: Priority,
    // Windows reads the peer through its pinned handle instead.
    #[cfg_attr(not(unix), allow(dead_code))]
    pub(crate) pid: Option<u32>,
}

impl Client {
    #[cfg(unix)]
    pub(crate) fn of_process(pid: u32) -> Self {
        Self {
            priority: Priority::of_process(pid),
            pid: Some(pid),
        }
    }
}

thread_local! {
    static INVOCATION: std::cell::Cell<Option<Client>> = const { std::cell::Cell::new(None) };
}

/// The priority of one synchronous request whose build is another process.
/// The daemon enters it with its client; a request outside any scope reads
/// its own process.
pub(crate) struct InvocationPriorityScope {
    previous: Option<Client>,
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl InvocationPriorityScope {
    /// A scope with `priority` and no client process: whether the build is
    /// held is read from this process.
    #[cfg(test)]
    pub(crate) fn enter(priority: Option<Priority>) -> Self {
        Self::enter_client(priority.map(|priority| Client {
            priority,
            pid: None,
        }))
    }

    pub(crate) fn enter_client(client: Option<Client>) -> Self {
        Self {
            previous: INVOCATION.with(|slot| slot.replace(client)),
            _thread: std::marker::PhantomData,
        }
    }
}

impl Drop for InvocationPriorityScope {
    fn drop(&mut self) {
        INVOCATION.with(|slot| slot.set(self.previous));
    }
}

/// The priority of the build this thread is serving, as it was when the
/// request arrived.
pub(crate) fn current() -> Priority {
    INVOCATION
        .with(std::cell::Cell::get)
        .map_or_else(Priority::of_this_process, |client| client.priority)
}

/// How the build this thread is serving stands now, read again on every
/// call: a waiting request is re-ranked from it each time it looks (see
/// [`super::queue`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Standing {
    pub(crate) priority: Priority,
    /// A process the request descends from is stopped (`SIGSTOP`, job
    /// control, or a scheduler above the build such as one that holds
    /// lower-ranked builds at their Cargo). Such a build starts no new work
    /// until it is continued, so its waiting requests yield to builds that
    /// run.
    pub(crate) held: bool,
}

pub(crate) fn standing() -> Standing {
    let client = INVOCATION.with(std::cell::Cell::get);
    #[cfg(unix)]
    {
        let pid = client
            .and_then(|client| client.pid)
            .unwrap_or_else(std::process::id);
        let priority = match client {
            Some(Client { pid: Some(pid), .. }) => Priority::of_process(pid),
            Some(client) => client.priority,
            None => Priority::of_this_process(),
        };
        Standing {
            priority,
            held: lineage_stopped(pid),
        }
    }
    #[cfg(not(unix))]
    {
        Standing {
            priority: client.map_or_else(Priority::of_this_process, |client| client.priority),
            held: false,
        }
    }
}

#[cfg(any(test, unix))]
/// The most processes [`lineage_stopped`] reads: a build sits a handful of
/// levels below its Cargo, and a cycle of reused parent IDs cannot loop.
const LINEAGE_DEPTH: usize = 32;

/// Whether `pid` or one of its ancestors is stopped. An unreadable process
/// ends the walk as not stopped.
#[cfg(unix)]
pub(crate) fn lineage_stopped(pid: u32) -> bool {
    lineage_stopped_by(pid, process_status)
}

/// [`lineage_stopped`] over `status`, which gives a process's stopped state
/// and parent.
#[cfg(any(test, unix))]
pub(crate) fn lineage_stopped_by(
    pid: u32,
    mut status: impl FnMut(u32) -> Option<(bool, u32)>,
) -> bool {
    let mut pid = pid;
    for _ in 0..LINEAGE_DEPTH {
        let Some((stopped, parent)) = status(pid) else {
            return false;
        };
        if stopped {
            return true;
        }
        if parent <= 1 || parent == pid {
            return false;
        }
        pid = parent;
    }
    false
}

/// A `/proc/<pid>/stat` line's stopped state and parent. The command name
/// can hold spaces and parentheses, so the fields follow the last `)`.
/// `T` is a stop; `t`, a tracer's stop, is not one a scheduler asked for.
#[cfg(any(test, target_os = "linux"))]
pub(crate) fn linux_stat_status(stat: &str) -> Option<(bool, u32)> {
    let mut fields = stat.rsplit_once(')')?.1.split_whitespace();
    let state = fields.next()?;
    let parent = fields.next()?.parse().ok()?;
    Some((state == "T", parent))
}

#[cfg(target_os = "linux")]
fn process_status(pid: u32) -> Option<(bool, u32)> {
    linux_stat_status(&std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?)
}

/// `SSTOP` in `<sys/proc.h>`.
#[cfg(any(test, target_os = "macos"))]
pub(crate) const MACOS_STOPPED: u32 = 4;

#[cfg(target_os = "macos")]
// Linux mutation shards cannot execute this macOS syscall adapter. The walk
// is exercised on every host; macOS CI also tests real stopped processes.
#[mutants::skip]
fn process_status(pid: u32) -> Option<(bool, u32)> {
    let pid = libc::c_int::try_from(pid).ok()?;
    // SAFETY: as in `process_is_background`.
    let mut info = unsafe { std::mem::zeroed::<libc::proc_bsdinfo>() };
    let size = std::mem::size_of::<libc::proc_bsdinfo>() as libc::c_int;
    let written =
        unsafe { libc::proc_pidinfo(pid, libc::PROC_PIDTBSDINFO, 0, (&raw mut info).cast(), size) };
    (written == size).then_some((info.pbi_status == MACOS_STOPPED, info.pbi_ppid))
}

#[cfg(all(unix, not(any(target_os = "macos", target_os = "linux"))))]
#[mutants::skip] // Not compiled by the supported-platform CI matrix.
fn process_status(pid: u32) -> Option<(bool, u32)> {
    let _ = pid;
    None
}

/// Whether macOS reports a process as background: Darwin background set from
/// inside (`PROC_FLAG_DARWINBG`) or outside (`PROC_FLAG_EXT_DARWINBG`), or
/// lowered with nice.
#[cfg(any(test, target_os = "macos"))]
pub(crate) fn macos_background(flags: u32, nice: i32) -> bool {
    const DARWIN_BACKGROUND_FLAGS: u32 = 0x0001_8000;
    flags & DARWIN_BACKGROUND_FLAGS != 0 || nice > 0
}

/// Whether Linux runs a process as background: lowered with nice, or under
/// the batch or idle policy.
#[cfg(any(test, target_os = "linux"))]
pub(crate) fn linux_background(nice: i32, policy: i32) -> bool {
    const SCHED_BATCH: i32 = 3;
    const SCHED_IDLE: i32 = 5;
    nice > 0 || policy == SCHED_BATCH || policy == SCHED_IDLE
}

/// Whether a Windows priority class is below normal.
#[cfg(any(test, windows))]
pub(crate) fn windows_background(class: u32) -> bool {
    const IDLE_PRIORITY_CLASS: u32 = 0x40;
    const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x4000;
    class == IDLE_PRIORITY_CLASS || class == BELOW_NORMAL_PRIORITY_CLASS
}

#[cfg(target_os = "macos")]
// Linux mutation shards cannot execute this macOS syscall adapter. The
// classifier is exercised on every host; macOS CI also tests real processes.
#[mutants::skip]
fn process_is_background(pid: u32) -> bool {
    let Ok(pid) = libc::c_int::try_from(pid) else {
        return false;
    };
    // Niceness remains readable when the wider process-info query is denied.
    // SAFETY: getpriority only reads the scheduling state of this process.
    let nice = unsafe { libc::getpriority(libc::PRIO_PROCESS, pid as u32) };
    // SAFETY: `proc_bsdinfo` is a C struct of integers and arrays; zero is a
    // valid value, and the kernel writes at most `size` bytes into it.
    let mut info = unsafe { std::mem::zeroed::<libc::proc_bsdinfo>() };
    let size = std::mem::size_of::<libc::proc_bsdinfo>() as libc::c_int;
    let written =
        unsafe { libc::proc_pidinfo(pid, libc::PROC_PIDTBSDINFO, 0, (&raw mut info).cast(), size) };
    let flags = if written == size { info.pbi_flags } else { 0 };
    macos_background(flags, nice)
}

#[cfg(target_os = "linux")]
fn process_is_background(pid: u32) -> bool {
    let Ok(sched_pid) = libc::pid_t::try_from(pid) else {
        return false;
    };
    // Both calls return -1 on failure: a raised nice value and no policy,
    // which reads as foreground.
    // SAFETY: both calls only read the scheduling state of `pid`.
    let nice = unsafe { libc::getpriority(libc::PRIO_PROCESS, pid) };
    // SAFETY: as above.
    let policy = unsafe { libc::sched_getscheduler(sched_pid) };
    linux_background(nice, policy)
}

#[cfg(all(unix, not(any(target_os = "macos", target_os = "linux"))))]
#[mutants::skip] // Not compiled by the supported-platform CI matrix.
fn process_is_background(pid: u32) -> bool {
    let _ = pid;
    false
}

#[cfg(windows)]
#[mutants::skip] // Windows API adapter; its class decoder is tested on every host.
fn this_process_is_background() -> bool {
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetPriorityClass};
    // SAFETY: the pseudo handle of this process needs no closing.
    windows_background(unsafe { GetPriorityClass(GetCurrentProcess()) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn macos_background_reads_both_darwin_flags_and_nice() {
        assert!(!macos_background(0, 0));
        assert!(macos_background(0x8000, 0), "set from inside");
        assert!(macos_background(0x10000, 0), "set by taskpolicy -b");
        assert!(macos_background(0, 1), "niced");
        assert!(!macos_background(0x21404010, 0), "a plain process");
        assert!(
            macos_background(0x21014010, 0),
            "the same after taskpolicy -b"
        );
        assert!(!macos_background(0, -5), "a raised priority is foreground");
    }

    #[test]
    fn linux_background_reads_nice_and_policy() {
        assert!(!linux_background(0, 0), "SCHED_OTHER at nice 0");
        assert!(linux_background(1, 0));
        assert!(!linux_background(-1, 0));
        assert!(linux_background(0, 3), "SCHED_BATCH");
        assert!(linux_background(0, 5), "SCHED_IDLE");
        assert!(!linux_background(0, 1), "SCHED_FIFO");
        assert!(!linux_background(0, 2), "SCHED_RR");
    }

    #[test]
    fn windows_background_is_idle_or_below_normal() {
        assert!(windows_background(0x40));
        assert!(windows_background(0x4000));
        assert!(!windows_background(0x20), "normal");
        assert!(!windows_background(0x8000), "above normal");
        assert!(!windows_background(0), "unreadable");
        assert_eq!(Priority::of_windows_class(0x40), Priority::Background);
        assert_eq!(Priority::of_windows_class(0x4000), Priority::Background);
        assert_eq!(Priority::of_windows_class(0x20), Priority::Foreground);
    }

    #[test]
    fn a_scope_replaces_the_process_until_dropped() {
        let own = Priority::of_this_process();
        assert_eq!(current(), own);
        {
            let _outer = InvocationPriorityScope::enter(Some(Priority::Background));
            assert_eq!(current(), Priority::Background);
            {
                let _inner = InvocationPriorityScope::enter(Some(Priority::Foreground));
                assert_eq!(current(), Priority::Foreground);
            }
            assert_eq!(current(), Priority::Background);
            let _none = InvocationPriorityScope::enter(None);
            assert_eq!(current(), own, "no client priority reads this process");
        }
        assert_eq!(current(), own);
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn niced_fixture_command(out: &std::path::Path) -> std::process::Command {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "scheduler::priority::tests::niced_process_fixture",
                "--ignored",
                "--quiet",
            ])
            .env("KACHE_TEST_PRIORITY_OUT", out)
            .env_remove("KACHE_TEST_PRIORITY_HOLD");
        command
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn a_niced_child_reads_as_background() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("priority");
        let mut niced = niced_fixture_command(&out)
            .env("KACHE_TEST_PRIORITY_HOLD", "1")
            .spawn()
            .unwrap();
        // The fixture publishes only after setting and checking its priority.
        // Keep it alive until the parent has queried that same process.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !out.exists() && std::time::Instant::now() < deadline {
            if niced.try_wait().unwrap().is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let ready = out.exists();
        let observed = Priority::of_process(niced.id());
        let _ = niced.kill();
        let _ = niced.wait();
        assert!(ready, "the child did not establish background priority");
        assert_eq!(observed, Priority::Background);
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    #[ignore = "subprocess fixture for this_process_reads_its_own_nice"]
    fn niced_process_fixture() {
        let out = std::env::var_os("KACHE_TEST_PRIORITY_OUT").expect("fixture output");
        // `nice -n` adds to the inherited value: a raised-priority runner can
        // remain foreground after +5 or +10. Only this child changes priority.
        // SAFETY: these calls read and lower the current process's priority.
        // On Linux niceness belongs to a thread. Use the process's main
        // thread, which both the parent and `of_this_process` query.
        let pid = std::process::id();
        let before = unsafe { libc::getpriority(libc::PRIO_PROCESS, pid) };
        let changed = unsafe { libc::setpriority(libc::PRIO_PROCESS, pid, 20) };
        assert_eq!(
            changed,
            0,
            "setting background nice from {before}: {}",
            std::io::Error::last_os_error()
        );
        let actual = unsafe { libc::getpriority(libc::PRIO_PROCESS, pid) };
        assert!(
            actual > 0,
            "fixture nice must be positive; before={before}, after={actual}"
        );
        let observed = (Priority::of_this_process(), current());
        assert_eq!(
            observed,
            (Priority::Background, Priority::Background),
            "nice before={before}, after={actual}"
        );
        std::fs::write(out, format!("{observed:?}")).unwrap();
        if std::env::var_os("KACHE_TEST_PRIORITY_HOLD").is_some() {
            std::thread::sleep(std::time::Duration::from_secs(20));
        }
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn this_process_reads_its_own_nice() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("priority");
        let status = niced_fixture_command(&out).status().unwrap();
        assert!(status.success());
        assert_eq!(
            std::fs::read_to_string(out).unwrap(),
            "(Background, Background)"
        );
    }

    #[test]
    fn a_lineage_is_stopped_when_any_ancestor_is() {
        // pid -> (stopped, parent)
        let tree = |stopped: &'static [u32]| {
            move |pid: u32| match pid {
                40 => Some((stopped.contains(&40), 30)),
                30 => Some((stopped.contains(&30), 20)),
                20 => Some((stopped.contains(&20), 1)),
                _ => None,
            }
        };
        assert!(!lineage_stopped_by(40, tree(&[])));
        assert!(lineage_stopped_by(40, tree(&[40])), "the process itself");
        assert!(lineage_stopped_by(40, tree(&[30])), "its parent");
        assert!(lineage_stopped_by(40, tree(&[20])), "a grandparent");
        assert!(
            !lineage_stopped_by(30, tree(&[40])),
            "a child is not an ancestor"
        );
        assert!(!lineage_stopped_by(99, tree(&[20])), "unreadable");
        // launchd or init ends the walk without being read.
        assert!(!lineage_stopped_by(20, |pid| match pid {
            20 => Some((false, 1)),
            _ => Some((true, 0)),
        }));
        // A process that is its own parent, or a cycle, ends the walk.
        assert!(!lineage_stopped_by(7, |_| Some((false, 7))));
        let mut reads = 0;
        assert!(!lineage_stopped_by(5, |pid| {
            reads += 1;
            Some((false, if pid == 5 { 6 } else { 5 }))
        }));
        assert_eq!(reads, LINEAGE_DEPTH);
        let mut reads = 0;
        assert!(
            lineage_stopped_by(2, |pid| {
                reads += 1;
                Some((reads == LINEAGE_DEPTH, pid + 1))
            }),
            "the deepest ancestor read counts"
        );
    }

    #[test]
    fn a_linux_stat_line_gives_the_stop_and_the_parent() {
        assert_eq!(linux_stat_status("42 (cargo) T 17 42"), Some((true, 17)));
        assert_eq!(linux_stat_status("42 (a) (b) S 9 1"), Some((false, 9)));
        assert_eq!(linux_stat_status("42 (gdb'd) t 9 1"), Some((false, 9)));
        assert_eq!(linux_stat_status("42 (x) R"), None);
        assert_eq!(linux_stat_status("42 (x) R x"), None);
        assert_eq!(linux_stat_status("no fields"), None);
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn a_stopped_parent_holds_its_running_child() {
        use std::io::BufRead;
        let mut parent = std::process::Command::new("/bin/sh")
            .args(["-c", "sleep 30 & echo $!; wait"])
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let mut line = String::new();
        std::io::BufReader::new(parent.stdout.take().unwrap())
            .read_line(&mut line)
            .unwrap();
        let child: u32 = line.trim().parse().unwrap();
        let parent_pid = parent.id() as libc::pid_t;
        let settled = |want: bool| {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while lineage_stopped(child) != want && std::time::Instant::now() < deadline {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            lineage_stopped(child)
        };
        let running = settled(false);
        unsafe { libc::kill(parent_pid, libc::SIGSTOP) };
        let stopped = settled(true);
        unsafe { libc::kill(parent_pid, libc::SIGCONT) };
        let continued = settled(false);
        unsafe {
            libc::kill(child as libc::pid_t, libc::SIGKILL);
            libc::kill(parent_pid, libc::SIGKILL);
        }
        let _ = parent.wait();
        assert!(!running, "a running lineage");
        assert!(stopped, "the parent was stopped");
        assert!(!continued, "the parent was continued");
    }

    #[cfg(unix)]
    #[test]
    fn a_client_scope_reads_its_process_again() {
        let own = standing();
        assert!(!own.held);
        {
            let _scope = InvocationPriorityScope::enter(Some(Priority::Background));
            assert_eq!(standing().priority, Priority::Background);
            assert!(!standing().held, "no client: this process");
        }
        let mut sleeper = std::process::Command::new("/bin/sleep")
            .arg("30")
            .spawn()
            .unwrap();
        let pid = sleeper.id();
        let client = Client::of_process(pid);
        assert_eq!(client.pid, Some(pid));
        let _scope = InvocationPriorityScope::enter_client(Some(Client {
            priority: Priority::Background,
            pid: Some(pid),
        }));
        assert_eq!(current(), Priority::Background, "as it arrived");
        assert_eq!(
            standing().priority,
            Priority::of_process(pid),
            "as it is now"
        );
        unsafe { libc::kill(pid as libc::pid_t, libc::SIGSTOP) };
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !standing().held && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let held = standing().held;
        let _ = sleeper.kill();
        let _ = sleeper.wait();
        assert!(held, "the client was stopped");
    }

    #[cfg(unix)]
    #[test]
    fn an_unreadable_process_is_foreground() {
        assert_eq!(Priority::of_process(u32::MAX), Priority::Foreground);
        assert_eq!(Priority::of_process(0x7fff_fff0), Priority::Foreground);
    }
}
