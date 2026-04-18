======================
TASK-NAME: test-param2
======================

Two parameters must be passed to this task: identified as {DIR} and {HOST}.

If no {DIR} parameter was provided:
 * Display the message: "{TASK-NAME}: missing required {DIR} parameter"
 * This task has failed, exit it now

If {DIR} is not a directory:
 * Display the message: "{TASK-NAME}: {DIR} is not a directory"
 * This task has failed: exit it now
 
If no {HOST} parameter was provided:
 * Display the message: "{TASK-NAME}: missing required {HOST} parameter"
 * This task has failed, exit it now

Display the message: "{TASK-NAME}: Executing in {DIR} for host: {HOST}"

This task has completed successfully.

