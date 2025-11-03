# Alembic Migrations

## Setup

Alembic is configured for this project to manage database schema migrations.

## Running Migrations

### Apply migrations
```bash
alembic upgrade head
```

### Rollback last migration
```bash
alembic downgrade -1
```

### Create new migration
```bash
alembic revision -m "description"
```

## Current Migration

### `a1b2c3d4_add_user_profile_and_memory_event`

**Tables created:**
- `userprofile` - User profile information with VIN, preferences, etc.
- `memoryevent` - Memory tracking and event logs

**Indexes:**
- `idx_memoryevent_user_id` on `memoryevent(user_id)`

