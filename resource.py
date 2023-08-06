class Resource:
    def __init__(self, memory, cores):
        self.memory = memory
        self.cores = cores

    def __str__(self):
        return f"R({self.memory},{self.cores})"

    def verify_task(self, task):
        return task.memory <= self.memory and task.cores <= self.cores

    def verify_task_or_throw(self, task, current_time):
        if not self.verify_task(task, current_time):
            raise Exception(f'{task} is not doable on {self}')

    def reserve(self, task, current_time):
        # self.verify_task_or_throw(task, current_time)
        self.memory -= task.memory
        self.cores -= task.cores

    def release(self, task):
        self.memory += task.memory
        self.cores += task.cores
