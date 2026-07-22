uint32 PROTOCOL_VERSION=1

uint8 STATUS_OK=0
uint8 STATUS_NOT_INITIALIZED=1
uint8 STATUS_INVALID_REQUEST=2
uint8 STATUS_STALE=3
uint8 STATUS_TIMEOUT=4
uint8 STATUS_CANCELED=5
uint8 STATUS_RESET_MISMATCH=6
uint8 STATUS_OBSERVATION_MISSING=7
uint8 STATUS_INTERNAL_ERROR=8
uint8 STATUS_SHUTTING_DOWN=9

uint8 LIFECYCLE_UNINITIALIZED=0
uint8 LIFECYCLE_LOADING=1
uint8 LIFECYCLE_READY=2
uint8 LIFECYCLE_RESETTING=3
uint8 LIFECYCLE_ERROR=4
uint8 LIFECYCLE_SHUTTING_DOWN=5

std_msgs/Header header
uint32 protocol_version
uint8 status_code
string status_message
uint8 lifecycle_state
bool initialized
string episode_id
uint64 reset_generation
uint64 last_sequence_id
bool safe_stop
string model_revision
string checkpoint_revision
