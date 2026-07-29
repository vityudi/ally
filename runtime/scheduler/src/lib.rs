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

    pub fn run_due(&self, events: &EventBus) {
        for task in &self.tasks {
            events.publish(Event::ReminderCreated {
                reminder_id: task.name.clone(),
            });
        }
    }
}
