//! A compile that is about to start waits while the store's volume is under
//! the job floor, three quarters of `cache.auto_recover_min_free_bytes`.
//!
//! Recovery starts evicting at the recovery floor, but only every five
//! minutes, for a few seconds, and never while another GC holds `gc.lock`,
//! and nothing else slows what builds write. Ten parallel cold builds write
//! faster than a size sweep evicts, so the volume filled to zero with the
//! store far over its cap. Between the recovery floor and this one builds run
//! and recovery evicts; under this one no new compile starts, so the builds in
//! flight finish and eviction catches up. A compile that has waited
//! [`MAX_WAIT`] runs anyway: a floor that space held outside the store keeps
//! breached must not hang every build.

use crate::config::Config;
use kache_fs::VolumeUsage;
use std::time::{Duration, Instant};

const POLL: Duration = Duration::from_secs(2);
const MAX_WAIT: Duration = Duration::from_secs(20 * 60);

#[cfg(test)]
thread_local! {
    /// The poll and longest wait of a hold on this thread. Other tests compile
    /// on volumes they fake as full, so a test hold gives up at once unless
    /// its test asks for longer.
    static TEST_TIMING: std::cell::Cell<(Duration, Duration)> =
        const { std::cell::Cell::new((Duration::from_millis(1), Duration::ZERO)) };
}

/// The poll and longest wait of a hold.
fn timing() -> (Duration, Duration) {
    #[cfg(test)]
    return TEST_TIMING.with(std::cell::Cell::get);
    #[cfg(not(test))]
    (POLL, MAX_WAIT)
}

/// The free bytes under which a new compile waits, for a recovery floor.
pub(crate) fn job_floor(recovery_floor: u64) -> u64 {
    recovery_floor - recovery_floor / 4
}

fn under(floor: u64, usage: Option<VolumeUsage>) -> bool {
    usage.is_some_and(|usage| floor <= usage.total && usage.free < floor)
}

/// How a hold ended.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Hold {
    /// The volume was over the floor: no wait.
    Clear,
    /// Waited until eviction freed space.
    Freed(Duration),
    /// Waited [`MAX_WAIT`] and ran anyway.
    GaveUp,
}

/// Wait while the store's volume is under the job floor. Never fails the
/// compile that asked.
pub(crate) fn hold_new_job(config: &Config) -> Hold {
    // Without auto-GC nothing evicts, so a hold would only wait out its time.
    let recovery_floor = if config.auto_gc {
        config.auto_recover_min_free_bytes
    } else {
        0
    };
    let (poll, max_wait) = timing();
    hold_with(
        job_floor(recovery_floor),
        polls(poll, max_wait),
        max_wait,
        || kache_fs::volume_usage(&config.cache_dir),
        || wait_step(config, poll),
        crate::notice::show,
    )
}

/// One wait of a held compile: run recovery if it is due, then sleep.
fn wait_step(config: &Config, poll: Duration) {
    crate::disk_recovery::run(config);
    std::thread::sleep(poll);
}

/// How many polls of `poll` fill `max_wait`, at least one. A count, not a
/// clock, bounds the wait, so no reading of the time can make it endless.
fn polls(poll: Duration, max_wait: Duration) -> u128 {
    (max_wait.as_millis() / poll.as_millis().max(1)).max(1)
}

fn hold_with(
    floor: u64,
    polls: u128,
    max_wait: Duration,
    mut usage: impl FnMut() -> Option<VolumeUsage>,
    mut wait: impl FnMut(),
    mut say: impl FnMut(&str),
) -> Hold {
    let first = usage();
    if !under(floor, first) {
        return Hold::Clear;
    }
    let free = first.map_or(0, |usage| usage.free);
    say(&format!(
        "kache: holding a compile: {} free on the cache volume, under the {} floor \
         (3/4 of cache.auto_recover_min_free_bytes); waiting up to {}m for eviction",
        crate::term::bytes(free),
        crate::term::bytes(floor),
        max_wait.as_secs() / 60,
    ));
    tracing::warn!(free, floor, "compile held under the disk floor");
    let started = Instant::now();
    for _ in 0..polls {
        wait();
        if !under(floor, usage()) {
            return Hold::Freed(started.elapsed());
        }
    }
    say(&format!(
        "kache: the cache volume is still under the {} floor after {}m; compiling anyway",
        crate::term::bytes(floor),
        max_wait.as_secs() / 60,
    ));
    Hold::GaveUp
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(free: u64) -> Option<VolumeUsage> {
        Some(VolumeUsage { total: 1000, free })
    }

    #[test]
    fn the_job_floor_is_three_quarters_of_the_recovery_floor() {
        assert_eq!(job_floor(0), 0);
        assert_eq!(job_floor(100), 75);
        assert_eq!(job_floor(128 << 30), 96 << 30);
        assert_eq!(job_floor(3), 3);
    }

    #[test]
    fn only_a_readable_volume_under_a_floor_it_can_reach_holds() {
        assert!(under(100, at(99)));
        assert!(!under(100, at(100)));
        assert!(!under(100, at(101)));
        assert!(!under(0, at(0)));
        assert!(!under(1001, at(10)));
        assert!(under(1000, at(10)));
        assert!(!under(100, None));
    }

    #[test]
    fn a_volume_over_the_floor_never_waits_or_speaks() {
        let mut waits = 0;
        let mut said = Vec::new();
        let hold = hold_with(
            100,
            600,
            MAX_WAIT,
            || at(500),
            || waits += 1,
            |line| said.push(line.to_owned()),
        );
        assert_eq!(hold, Hold::Clear);
        assert_eq!(waits, 0);
        assert!(said.is_empty());
    }

    #[test]
    fn a_held_compile_starts_once_eviction_frees_space() {
        let mut free = [10, 20, 150].into_iter();
        let mut waits = 0;
        let mut said = Vec::new();
        let hold = hold_with(
            100,
            600,
            MAX_WAIT,
            || at(free.next().unwrap()),
            || waits += 1,
            |line| said.push(line.to_owned()),
        );
        assert!(matches!(hold, Hold::Freed(_)));
        assert_eq!(waits, 2);
        assert_eq!(said.len(), 1);
        assert!(said[0].contains("holding a compile"), "{said:?}");
        assert!(said[0].contains("20m"), "{said:?}");
    }

    #[test]
    fn a_floor_nothing_frees_gives_up_after_the_wait() {
        let mut waits = 0;
        let mut said = Vec::new();
        let hold = hold_with(
            100,
            3,
            MAX_WAIT,
            || at(10),
            || waits += 1,
            |line| said.push(line.to_owned()),
        );
        assert_eq!(hold, Hold::GaveUp);
        assert_eq!(waits, 3);
        assert!(said[1].contains("after 20m"), "{said:?}");
        assert_eq!(said.len(), 2);
        assert!(said[1].contains("compiling anyway"), "{said:?}");
    }

    #[test]
    fn a_held_test_compile_gives_up_at_once_by_default() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = crate::test_support::test_config(dir.path().join("cache"));
        config.auto_recover_min_free_bytes = 400;
        let _volume = kache_fs::testutil::VolumeUsageGuard::new(
            &config.cache_dir,
            Some(VolumeUsage {
                total: 1000,
                free: 0,
            }),
        );
        assert_eq!(hold_new_job(&config), Hold::GaveUp);
        assert_eq!(POLL, Duration::from_secs(2));
        assert_eq!(MAX_WAIT, Duration::from_secs(1200));
    }

    #[test]
    fn the_wait_is_a_count_of_polls() {
        assert_eq!(polls(POLL, MAX_WAIT), 600);
        assert_eq!(
            polls(Duration::from_millis(3), Duration::from_millis(10)),
            3
        );
        assert_eq!(polls(Duration::from_millis(1), Duration::ZERO), 1);
        assert_eq!(polls(Duration::ZERO, Duration::from_millis(5)), 5);
    }

    #[test]
    fn a_disabled_recovery_floor_never_holds() {
        let dir = tempfile::tempdir().unwrap();
        let config = crate::test_support::test_config(dir.path().join("cache"));
        assert_eq!(config.auto_recover_min_free_bytes, 0);
        let _volume = kache_fs::testutil::VolumeUsageGuard::new(
            &config.cache_dir,
            Some(VolumeUsage {
                total: 1000,
                free: 0,
            }),
        );
        assert_eq!(hold_new_job(&config), Hold::Clear);
    }

    #[test]
    fn a_compile_under_the_floor_waits_until_space_comes_back() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = crate::test_support::test_config(dir.path().join("cache"));
        config.auto_recover_min_free_bytes = 400;
        let volume = kache_fs::testutil::VolumeUsageGuard::new(
            &config.cache_dir,
            Some(VolumeUsage {
                total: 1000,
                free: 0,
            }),
        );
        // Dropping the fixture hands back the real volume, far over 300 bytes free.
        let freer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            drop(volume);
        });
        let poll = Duration::from_millis(20);
        TEST_TIMING.with(|timing| timing.set((poll, Duration::from_secs(60))));
        let hold = hold_new_job(&config);
        freer.join().unwrap();
        assert!(
            matches!(hold, Hold::Freed(waited) if waited >= poll),
            "{hold:?}"
        );
    }

    #[test]
    fn without_auto_gc_nothing_holds() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = crate::test_support::test_config(dir.path().join("cache"));
        config.auto_recover_min_free_bytes = 400;
        config.auto_gc = false;
        let _volume = kache_fs::testutil::VolumeUsageGuard::new(
            &config.cache_dir,
            Some(VolumeUsage {
                total: 1000,
                free: 0,
            }),
        );
        assert_eq!(hold_new_job(&config), Hold::Clear);
    }

    #[test]
    #[cfg(unix)]
    fn each_wait_runs_recovery() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = crate::test_support::test_config(dir.path().join("cache"));
        config.auto_recover_min_free_bytes = 400;
        let store = crate::store::Store::open(&config).unwrap();
        let _volume = kache_fs::testutil::VolumeUsageGuard::new(
            dir.path(),
            Some(VolumeUsage {
                total: 1000,
                free: 10,
            }),
        );
        let source = dir.path().join("old.rlib");
        std::fs::write(&source, "old").unwrap();
        store
            .put(
                "old",
                "old",
                &["lib".into()],
                &[],
                "host",
                "dev",
                &[(source.clone(), "old.rlib".into())],
                "",
                "",
            )
            .unwrap();
        store.remove_clone_for_test(&source);
        store.set_last_accessed_for_test("old", "-1 hour");
        wait_step(&config, Duration::ZERO);
        assert!(!store.contains("old"), "the wait evicted under pressure");
    }
}
