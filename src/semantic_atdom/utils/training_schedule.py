"""Pure helpers defining the progressive optimization cadence."""


def progressive_stage_iterations(
    *,
    new_image_count: int,
    transition_index: int,
    local_iterations_per_image: int,
    global_interval: int,
    global_iterations: int,
):
    values = (
        new_image_count,
        transition_index,
        local_iterations_per_image,
        global_interval,
        global_iterations,
    )
    if any(int(value) < 0 for value in values):
        raise ValueError("training schedule values must be non-negative")
    if int(transition_index) < 1 or int(global_interval) < 1:
        raise ValueError("transition_index and global_interval must be positive")
    local = int(new_image_count) * int(local_iterations_per_image)
    global_count = (
        int(global_iterations)
        if int(transition_index) % int(global_interval) == 0
        else 0
    )
    return local, global_count


def total_training_iterations(
    *,
    initial_iterations: int,
    final_iterations: int,
    stage_new_image_counts,
    local_iterations_per_image: int,
    global_interval: int,
    global_iterations: int,
) -> int:
    total = int(initial_iterations) + int(final_iterations)
    for index, count in enumerate(stage_new_image_counts, start=1):
        local, global_count = progressive_stage_iterations(
            new_image_count=int(count),
            transition_index=index,
            local_iterations_per_image=local_iterations_per_image,
            global_interval=global_interval,
            global_iterations=global_iterations,
        )
        total += local + global_count
    return total
