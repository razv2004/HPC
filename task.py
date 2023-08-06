class Task:
    def __init__(self, memory, cores, penalty, duration, delay, user):
        self.memory = memory
        self.cores = cores
        self.penalty = penalty
        self.duration = duration
        self.delay = delay
        self.user = user

    def __init__(self, duration_memory_cores, penalty, delay, user):
        self.duration = duration_memory_cores[0]
        self.memory = duration_memory_cores[1]
        self.cores = duration_memory_cores[2]
        self.penalty = penalty
        self.delay = delay
        self.user = user

    def get_task_score(self, time):
        time_difference = self.delay - time
        score = self.penalty * 1.005 ** time_difference
        return score

    def __str__(self):
        return f"T({self.memory},{self.cores},{self.penalty},{self.duration},{self.delay},{self.user.name})"
