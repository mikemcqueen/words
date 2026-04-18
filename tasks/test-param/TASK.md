=====================
TASK-NAME: test-param
=====================

A parameter must be passed to this task: identify it as {DIR}.

If no {DIR} parameter was provided:
 * Display the message: "{TASK-NAME}: missing required parameter"
 * This task has failed, exit it now

If {DIR} is not a directory:
 * Display the message: "{TASK-NAME}: {DIR} is not a directory"
 * This task has failed: exit it now

Display the message: "{TASK-NAME}: {DIR} is a directory"

Run the task ../tasks/test-param-pass/TASK.md {DIR}

If that task fails:
 * Display the message: "{TASK-NAME} test-param-pass failed"
 * This task has failed: exit it now

Display the message: "{TASK-NAME}: SUCCESS!"

This task has completed successfully.
