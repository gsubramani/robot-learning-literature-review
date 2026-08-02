---
name: ai-task
description: how to use the .ai folder for tasks
---

For generic tasks that require planning, intermediate script evaluation or accumulation of data without modfying the rest of the codebase. 

create a folder in the `.ai/tasks/task-<date-time>-<name>`

the task folder will contain the following subfolders

.ai/tasks/task-<date-time>-<name>/sources/
location where the sources go if there are any

.ai/tasks/task-<date-time>-<name>/scripts/
location where any python scripts would be present

.ai/tasks/task-<date-time>-<name>/docs/
location for documentation needed for the task/


.ai/tasks/task-<date-time>-<name>/plan/
location where plans are saved. use the interactive-planning skill for this: .devin\skills\interactive-planning

.ai/tasks/task-<date-time>-<name>/summary.md
a brief summary of the task that's being worked on. This would be updated once the task is complete too



# ENSURE YOU FOLLOW THE FOLDER NAMING CONVENTION task-<date-time>-<name>