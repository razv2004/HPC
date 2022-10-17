from abc import abstractmethod
import itertools
import random


class Resource:
    def __init__(self, cores, parameters=None):
        self.cores = cores
        self.parameters = [] if parameters is None else parameters

    def is_task_doable(self, task):
        return task.parameter is None or task.parameter in self.parameters

    def task_time(self, task):
        if not self.is_task_doable(task):
            raise Exception(f'Task ({task.required}) is not doable on Resource({self.cores})')
        return task.required / self.cores


class Task:
    def __init__(self, required, parameter=None):
        self.required = required
        self.parameter = parameter


class Cluster:
    def __init__(self, resources, tasks):
        self.resources = resources
        self.tasks = tasks
        self.queue = {}


class Scheduler:
    def __init__(self, cluster):
        self.cluster = cluster
        self.time = 0

    @abstractmethod
    def assign_next(self):
        pass

    def run(self):
        while len(self.cluster.tasks) > 0 or len(self.cluster.queue) > 0:
            resource, task = self.assign_next()
            if resource is None:
                self.move_time()
                continue
            self.cluster.resources.remove(resource)
            self.cluster.tasks.remove(task)
            duration = resource.task_time(task)
            if duration not in self.cluster.queue:
                self.cluster.queue[duration] = set()
            self.cluster.queue[duration].add(tuple([resource, task]))
            print(f'Adding Task ({task.required}) to Resource({resource.cores})')

    def move_time(self):
        next_checkpoint = sorted(self.cluster.queue.items())[0]
        del self.cluster.queue[next_checkpoint[0]]
        time = next_checkpoint[0]
        print(f'Time moved from {self.time} to {time}')
        self.time = time
        for finished_tasks in next_checkpoint[1]:
            resource = finished_tasks[0]
            task = finished_tasks[1]
            self.cluster.resources.append(resource)
            print(f'Done Task ({task.required}) on Resource({resource.cores})')


class RandomScheduler(Scheduler):
    def assign_next(self):
        if len(self.cluster.tasks) <= 0 or len(self.cluster.resources) <= 0:
            return None, None
        tasks_perm = self.pick_random_perm(self.cluster.tasks)
        resources_perm = self.pick_random_perm(self.cluster.resources)
        for task in tasks_perm:
            for resource in resources_perm:
                if resource.is_task_doable(task):
                    return resource, task
        return None, None

    @staticmethod
    def pick_random_perm(options):
        all_task_perm = list(itertools.permutations(options))
        k = random.randint(0, len(all_task_perm) - 1)
        chosen_perm = all_task_perm[k]
        return chosen_perm


if __name__ == '__main__':
    r1 = Resource(10)
    r2 = Resource(2)
    r3 = Resource(5)

    t1 = Task(3)
    t2 = Task(7)
    t3 = Task(50)

    c = Cluster([r1, r2, r3], [t1, t2, t3])

    random_scheduler = RandomScheduler(c)

    random_scheduler.run()
