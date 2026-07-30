//! Scheduler: autonomous, time-based execution (e.g. daily briefings,
//! reminders) that runs without a direct user request.

use ally_events::{Event, EventBus};

pub struct ScheduledTask {
    pub name: String,
}

#[derive(Default)]
pub struct Scheduler {
    tasks: Vec<ScheduledTask>,
}

impl Scheduler {
    pub fn new() -> Self {
        Self { tasks: Vec::new() }
    }

    pub fn add_task(&mut self, task: ScheduledTask) {
        self.tasks.push(task);
    }

    /// Number of tasks currently queued, run or not.
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    /// Whether any task is currently queued.
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    pub fn run_due(&self, events: &EventBus) {
        for task in &self.tasks {
            events.publish(Event::ReminderCreated {
                reminder_id: task.name.clone(),
            });
        }
    }
}
