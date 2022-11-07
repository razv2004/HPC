import copy
from abc import abstractmethod
import itertools
import random
from queue import PriorityQueue
from datetime import datetime
import numpy

EPSILON = 0.00000001


class Task:
    def __init__(self, cores, memory, score, duration, parameter=None):
        self.cores = cores
        self.memory = memory
        self.score = score
        self.duration = duration
        self.parameter = parameter

    def __str__(self):
        return f"T({self.cores},{self.memory},{self.score},{self.duration})"


class Resource:
    def __init__(self, cores, memory, parameters=None):
        self.cores = cores
        self.memory = memory
        self.parameters = [] if parameters is None else parameters

    def __str__(self):
        return f"R({self.cores},{self.memory})"

    def verify_parameters(self, parameter):
        return parameter is None or parameter in self.parameters

    def verify_task(self, task):
        return self.verify_parameters(task.parameter) and task.cores <= self.cores and task.memory <= self.memory

    def verify_task_or_throw(self, task):
        if not self.verify_task(task):
            raise Exception(f'{task} is not doable on {self}')

    def run_task(self, task):
        self.verify_task_or_throw(task)
        self.cores -= task.cores
        self.memory -= task.memory

    def release_task(self, task):
        self.cores += task.cores
        self.memory += task.memory


class Cluster:
    def __init__(self, resources, tasks):
        self.resources = copy.deepcopy(resources)
        self.tasks = copy.deepcopy(tasks)
        self.task_index = 0
        self.queue = PriorityQueue()

    def __str__(self):
        resources = ','.join([r.__str__() for r in self.resources])
        tasks = ','.join([t.__str__() for t in self.tasks])
        queue = self.queue.queue.__str__()
        return f"Cluster({resources} --- {tasks} --- {queue})"

    def run_task(self, current_time, resource, task):
        self.task_index += 1
        resource.run_task(task)
        finish_time = current_time + task.duration
        self.queue.put((finish_time, self.task_index, (resource, task)))
        self.tasks.remove(task)

    def is_done(self):
        return len(self.tasks) == 0 and self.queue.qsize() == 0


class Scheduler:
    def __init__(self, cluster):
        self.cluster = cluster
        self.time = 0
        self.score = 0

    @abstractmethod
    def assign_next(self):
        pass

    def run(self):
        while not self.cluster.is_done():
            resource, task = self.assign_next()
            if resource is None or task is None:
                if self.cluster.queue.qsize() == 0:
                    break
                self.move_time()
                continue
            self.cluster.run_task(self.time, resource, task)
        return self.score

    def move_time(self):
        original = self.time

        while self.time == original and not self.cluster.queue.qsize() == 0:
            next_item = self.cluster.queue.get()
            if self.time == original:
                self.time = next_item[0]
            elif self.time != next_item[0]:
                self.cluster.queue.put(next_item)
                return
            resource, task = next_item[2]
            resource.release_task(task)
            self.score += task.score * 0.9 ** self.time


class GreedyScheduler(Scheduler):
    def assign_next(self):
        if len(self.cluster.tasks) <= 0 or len(self.cluster.resources) <= 0:
            return None, None
        for task in sorted(self.cluster.tasks, key=lambda t: t.score/(t.duration**2 + EPSILON), reverse=True):
            for resource in sorted(self.cluster.resources, key=lambda r: 2*r.cores + r.memory):
                if resource.verify_task(task):
                    return resource, task
        return None, None


class RandomScheduler(Scheduler):
    def assign_next(self):
        if len(self.cluster.tasks) <= 0 or len(self.cluster.resources) <= 0:
            return None, None
        for _ in range(len(self.cluster.tasks)):
            task = random.choice(self.cluster.tasks)
            for _ in range(len(self.cluster.resources)):
                resource = random.choice(self.cluster.resources)
                if resource.verify_task(task):
                    return resource, task
        return None, None


class ExhaustiveRandomScheduler(Scheduler):
    def assign_next(self):
        if len(self.cluster.tasks) <= 0 or len(self.cluster.resources) <= 0:
            return None, None
        tasks_perm = self.pick_random_perm(self.cluster.tasks)
        resources_perm = self.pick_random_perm(self.cluster.resources)
        for task in tasks_perm:
            for resource in resources_perm:
                if resource.verify_task(task):
                    return resource, task
        return None, None

    @staticmethod
    def pick_random_perm(options):
        all_task_perm = list(itertools.permutations(options))
        k = random.randint(0, len(all_task_perm) - 1)
        chosen_perm = all_task_perm[k]
        return chosen_perm


def run_scheduler(r, t, scheduler, n):
    scores = []
    for i in range(n):
        random.seed(i)
        c = Cluster(r, t)
        scores.append(scheduler(c).run())
    print(f"{scheduler.__name__} --> {min(scores)}, {numpy.average(scores)}, {max(scores)}")


if __name__ == '__main__':
    random.seed(8192)
    r0 = [Resource(random.randint(0, 100), random.randint(0, 100)) for _ in range(30)]
    t0 = [Task(random.randint(0, 100), random.randint(0, 50), random.randint(0, 50), random.randint(0, 100)) for _ in range(300)]

    run_scheduler(r0, t0, GreedyScheduler, 1)
    run_scheduler(r0, t0, RandomScheduler, 30)

