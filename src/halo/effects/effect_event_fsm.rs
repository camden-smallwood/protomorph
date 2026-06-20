//! Effect event-lifecycle state machine — a faithful port of the engine's
//! sequential event chain (dllcache):
//!
//! - `effect_update_time @0x180309870` — walks the event chain each frame.
//! - `event_update_time   @0x180309620` — per-event delay→duration FSM.
//! - `effect_start_event  @0x1803063e0` — randomizes delay then duration (LCG),
//!   fires the initial burst.
//! - `effect_get_next_event_index @0x1803067a0` — sequential advance with
//!   `skip_fraction` skip + `loop_start_event` wrap.
//!
//! protomorph flattens `events[*].particle_systems[*]` into `LoadedEffect.systems`
//! (each tagged with `event_index`); this FSM decides, per frame, which event is
//! active and when each event fires its emitters' `starting_count` burst. The
//! per-emitter spawn math stays in [`super::particle_emitter`].

use blam_tags::effect::EffectDefinition;

/// Per-event runtime timing (mirrors the engine event_datum's `m_time` +0x0C /
/// target +0x10 / started+finished state byte +0x02).
#[derive(Debug, Clone, Default)]
pub struct EventRuntime {
    /// Accumulated time in the current phase.
    pub m_time: f32,
    /// Target time for the current phase: the (randomized) `delay` while not
    /// `started`, then the (randomized) `duration` once started.
    pub target: f32,
    /// `false` = delay phase (waiting to fire), `true` = duration phase (emitting).
    pub started: bool,
    /// The event has run to completion this cycle (advanced past).
    pub finished: bool,
}

/// Per-event, per-frame emission directive produced by the FSM tick.
#[derive(Debug, Clone, Copy, Default)]
pub struct EventFire {
    /// The event is in its duration window this frame → emitters emit `rate*dt`.
    pub active: bool,
    /// The delay→duration edge fired this frame → emitters add `starting_count`.
    pub fire_burst: bool,
}

/// Engine LCG step (`effect_start_event`: `seed = 1664525*seed + 1013904223`,
/// value = `HIWORD(seed) * 0.000015259022` ≈ `[0,1)`).
fn lcg_unit(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    ((*seed >> 16) & 0xffff) as f32 * 0.000015259022
}

/// Engine `real_random_range` — used for delay/duration bounds.
fn random_range(seed: &mut u32, lo: f32, hi: f32) -> f32 {
    if hi == lo {
        lo
    } else {
        lo + lcg_unit(seed) * (hi - lo)
    }
}

/// `effect_should_skip_event` — random skip by the event's `skip_fraction`.
fn should_skip(seed: &mut u32, skip_fraction: f32) -> bool {
    skip_fraction > 0.0 && lcg_unit(seed) < skip_fraction
}

/// `effect_get_next_event_index @0x1803067a0` — sequential advance, skipping
/// per `skip_fraction`, wrapping to `loop_start_event` at the end. Returns the
/// next event index, or `-1` when the (non-looping) chain has ended.
fn next_event_index(current: i32, seed: &mut u32, def: &EffectDefinition) -> i32 {
    let n = def.events.len() as i32;
    if n == 0 {
        return -1;
    }
    let loop_start = def.loop_start_event as i32;
    let mut idx = current;
    // Bounded — `skip_fraction` could in theory skip every event; the engine
    // relies on the chain terminating, we cap to one full lap to be safe.
    for _ in 0..(n + 1) {
        if idx >= n - 1 {
            if loop_start >= 0 && loop_start < n {
                idx = loop_start;
            } else {
                return -1; // chain ended, no loop
            }
        } else {
            idx += 1;
        }
        if idx >= 0 && idx < n && !should_skip(seed, def.events[idx as usize].skip_fraction) {
            return idx;
        }
    }
    -1
}

/// Enter `event_idx` as the current event: reset its runtime to the delay phase
/// with a freshly randomized delay target (`effect_start_event` phase 1 setup).
fn enter_event(rt: &mut EventRuntime, seed: &mut u32, def: &EffectDefinition, event_idx: usize) {
    let ev = &def.events[event_idx];
    rt.m_time = 0.0;
    rt.started = false;
    rt.finished = false;
    rt.target = random_range(seed, ev.delay_bounds.lower, ev.delay_bounds.upper);
}

/// Initialise an instance's event cursor at spawn — start at the first
/// non-skipped event with its delay armed. Returns the starting cursor (`-1`
/// when the effect has no events).
pub fn init_cursor(
    event_runtimes: &mut [EventRuntime],
    seed: &mut u32,
    def: &EffectDefinition,
) -> i32 {
    if def.events.is_empty() {
        return -1;
    }
    enter_event(&mut event_runtimes[0], seed, def, 0);
    0
}

/// Tick the SEQUENTIAL event chain (`effect_update_time` → `event_update_time`),
/// consuming `dt` across event transitions. Returns a per-event [`EventFire`]
/// directive for this frame (index = event index). When the chain ends without
/// a loop, `*current_event` is set to `-1` (caller may retire the instance).
pub fn tick_sequential(
    event_runtimes: &mut [EventRuntime],
    current_event: &mut i32,
    seed: &mut u32,
    def: &EffectDefinition,
    dt: f32,
) -> Vec<EventFire> {
    let n = def.events.len();
    let mut fires = vec![EventFire::default(); n];
    if dt <= 0.0 {
        return fires;
    }
    let mut remaining = dt;
    // Bound the per-frame transitions so a pathological all-zero
    // (delay 0 + duration 0) chain can't spin forever.
    for _ in 0..64 {
        if *current_event < 0 || remaining <= 0.0 {
            break;
        }
        let ev = *current_event as usize;
        if ev >= n {
            *current_event = -1;
            break;
        }
        let rt = &mut event_runtimes[ev];
        let phase_left = rt.target - rt.m_time;
        if remaining < phase_left {
            // Stays in the current phase this frame.
            rt.m_time += remaining;
            remaining = 0.0;
            if rt.started {
                fires[ev].active = true; // emitting through its duration
            }
            break;
        }
        // Current phase completes; carry the leftover dt forward.
        remaining -= phase_left.max(0.0);
        if !rt.started {
            // Delay elapsed → start the event (phase 2): randomize duration,
            // fire the burst, emit this frame.
            let dur = {
                let bounds = def.events[ev].duration_bounds;
                random_range(seed, bounds.lower, bounds.upper)
            };
            let rt = &mut event_runtimes[ev];
            rt.started = true;
            rt.m_time = 0.0;
            rt.target = dur;
            fires[ev].fire_burst = true;
            fires[ev].active = true;
            if dur > 0.0 {
                // Will keep emitting next frames until duration elapses.
                continue;
            }
            // Zero-duration burst: completes immediately, fall through to advance.
        }
        // Duration phase complete → advance to the next event.
        event_runtimes[ev].finished = true;
        let next = next_event_index(*current_event, seed, def);
        if next < 0 {
            *current_event = -1;
            break;
        }
        *current_event = next;
        enter_event(&mut event_runtimes[next as usize], seed, def, next as usize);
    }
    fires
}

/// Tick PARALLEL events (`RunEventsInParallel`) — every event runs its own
/// delay→duration cycle independently and self-restarts (continuous effects).
/// Each event is treated as a one-event sequential chain that loops onto itself.
pub fn tick_parallel(
    event_runtimes: &mut [EventRuntime],
    seed: &mut u32,
    def: &EffectDefinition,
    dt: f32,
) -> Vec<EventFire> {
    let n = def.events.len();
    let mut fires = vec![EventFire::default(); n];
    if dt <= 0.0 {
        return fires;
    }
    for ev in 0..n {
        let mut remaining = dt;
        for _ in 0..64 {
            if remaining <= 0.0 {
                break;
            }
            let rt = &mut event_runtimes[ev];
            // Arm the delay on first use (target 0 + not started + fresh).
            if !rt.started && rt.target == 0.0 && rt.m_time == 0.0 && !rt.finished {
                enter_event(rt, seed, def, ev);
            }
            let phase_left = rt.target - rt.m_time;
            if remaining < phase_left {
                rt.m_time += remaining;
                remaining = 0.0;
                if rt.started {
                    fires[ev].active = true;
                }
                break;
            }
            remaining -= phase_left.max(0.0);
            if !rt.started {
                let dur = {
                    let bounds = def.events[ev].duration_bounds;
                    random_range(seed, bounds.lower, bounds.upper)
                };
                let rt = &mut event_runtimes[ev];
                rt.started = true;
                rt.m_time = 0.0;
                rt.target = dur;
                fires[ev].fire_burst = true;
                fires[ev].active = true;
                if dur > 0.0 {
                    continue;
                }
            }
            // Completed → restart this event's own cycle (re-arm delay).
            enter_event(&mut event_runtimes[ev], seed, def, ev);
        }
    }
    fires
}
