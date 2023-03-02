import copy
import math

import matplotlib.pyplot as plt
from abc import abstractmethod
import itertools
import random
from queue import PriorityQueue
import numpy as np

RESOURCES_MEMORY_AND_CORES = [[1024, 48], [768, 48], [512, 32], [256, 32], [196, 24], [168, 24], [128, 24], [96, 16], [72, 8], [64, 8], [32, 4]]
TASK_DURATION_DISTRIBUTION = [[1, 60, 100, 300, 1000, 3000], [0.2, 0.25, 0.12, 0.13, 0.2, 0.1]]
TASK_PENALTY_DISTRIBUTION = [[1, 100, 10000], [0.8, 0.1, 0.1]]
TASK_MEMORIES_DISTRIBUTION = [[1, 2, 3, 4, 8, 16, 24], [0.5, 0.2, 0.1, 0.1, 0.02, 0.05, 0.03]]
COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']

U = 3  # number of users
R = 50  # number of resources
T = 20000  # number of tasks
C = 4  # cores per task
S = 100  # share range


class User:
    def __init__(self, name, share):
        self.name = name
        self.share = share

    def __str__(self):
        return f"U({self.name},{self.share})"


class Task:
    def __init__(self, memory, cores, penalty, duration, delay, user):
        self.memory = memory
        self.cores = cores
        self.penalty = penalty
        self.duration = duration
        self.delay = delay
        self.user = user

    def get_task_score(self, time):
        time_difference = self.delay - time
        score = self.penalty * 1.1 ** time_difference
        return score

    def __str__(self):
        return f"T({self.memory},{self.cores},{self.penalty},{self.duration},{self.delay},{self.user.name})"


class Resource:
    def __init__(self, memory, cores):
        self.memory = memory
        self.cores = cores

    def __str__(self):
        return f"R({self.memory},{self.cores})"

    def verify_task(self, task, current_time):
        return task.memory <= self.memory and task.cores <= self.cores and task.delay <= current_time

    def verify_task_or_throw(self, task, current_time):
        if not self.verify_task(task, current_time):
            raise Exception(f'{task} is not doable on {self}')

    def reserve_task(self, task, current_time):
        self.verify_task_or_throw(task, current_time)
        self.memory -= task.memory
        self.cores -= task.cores

    def release_task(self, task):
        # print(f'Done with {task.__str__()}')
        self.memory += task.memory
        self.cores += task.cores


class Cluster:
    def __init__(self, resources, tasks, users):
        self.resources = copy.deepcopy(resources)
        self.tasks = copy.deepcopy(tasks)
        self.users = copy.deepcopy(users)
        self.available_memory_per_hour = sum([resource.memory for resource in self.resources])
        self.available_cores_per_hour = sum([resource.cores for resource in self.resources])
        self.users_share = self.calculate_users_share()
        self.left_tasks = len(self.tasks)
        self.queue = PriorityQueue()
        self.current_running = {}

    def calculate_users_share(self):
        fair_users_share = {}
        for user in self.users:
            fair_users_share[user.name] = [user.share, 0, 0, 0]  # 0s represent no core time so far nor penalty so far
        return fair_users_share

    def __str__(self):
        resources = ','.join([r.__str__() for r in self.resources])
        tasks = ','.join([t.__str__() for t in self.tasks])
        queue = self.queue.queue.__str__()
        return f"Cluster({resources} --- {tasks} --- {queue})"

    def start_task(self, current_time, resource, task):
        self.left_tasks -= 1
        resource.reserve_task(task, current_time)
        finish_time = current_time + task.duration
        self.queue.put((finish_time, self.left_tasks, (resource, task)))
        self.tasks.remove(task)

    def get_user_tasks(self, user, time):
        return [task for task in self.tasks if task.user.name == user.name and task.delay <= time]

    def get_prioritized_users(self):
        return sorted(self.users, key=lambda u: self.users_share[u.name][1] / self.users_share[u.name][0])

    def is_done(self):
        return self.left_tasks == 0 and self.queue.qsize() == 0


class Scheduler:
    def __init__(self, cluster):
        self.cluster = cluster
        self.statistics = {}  # user --> statistics
        for u in self.cluster.users:
            self.statistics[u.name] = [[], [], [], []]
        self.checkpoints = []
        self.time = 0

        self.user_color = {}
        for i in range(len(self.cluster.users)):
            self.user_color[self.cluster.users[i].name] = COLORS[i]

    @abstractmethod
    def assign_next(self):
        pass

    def run(self):
        while not self.cluster.is_done():
            # print(f'{self.time=}, {self.cluster.left_tasks=}, {self.cluster.queue.qsize()=}')
            if self.cluster.left_tasks == 0 or self.are_memory_and_cores_insufficient():
                self.move_time_append_statistics()
                continue
            resource, task = self.assign_next()
            if resource is None or task is None:
                # if self.cluster.queue.qsize() == 0:
                #     break
                self.move_time_append_statistics()
                continue
            self.cluster.users_share[task.user.name][3] += task.get_task_score(self.time)
            self.cluster.start_task(self.time, resource, task)
        if self.cluster.queue.qsize() > 0:
            print(f"There are {self.cluster.queue.qsize()} unfinished tasks")

        self.print_statistics()

    def are_memory_and_cores_insufficient(self):
        max_available_memory, max_available_cores = self.max_memory_cores_available()
        min_required_memory, min_required_cores = self.min_memory_cores_required()
        insufficient_memory_and_cores = max_available_memory < min_required_memory or max_available_cores < min_required_cores
        return insufficient_memory_and_cores

    def append_users_statistics(self, time_difference):
        for user in self.cluster.users:
            user_share = self.cluster.users_share[user.name]
            self.statistics[user.name][0].append(user_share[0])
            self.statistics[user.name][1].append(user_share[1] / (self.cluster.available_memory_per_hour * self.time))
            self.statistics[user.name][2].append(user_share[2] / (self.cluster.available_cores_per_hour * self.time))
            # self.statistics[user.name][3].append(math.log(user_share[3] + 1))
            self.statistics[user.name][3].append(user_share[3])
        self.checkpoints.append(time_difference)

    def print_statistics(self):
        x = np.cumsum(self.checkpoints)

        figure, (ax1, ax2) = plt.subplots(2)

        for user in self.cluster.users:
            ax1.plot(x, self.statistics[user.name][0], self.user_color[user.name])
            ax1.plot(x, self.statistics[user.name][2], self.user_color[user.name])
            # ax1.set_title(f'{self.__class__.__name__},{min(self.statistics[user.name][3])},{max(self.statistics[user.name][3])}')
            ax1.set_title(f'{self.__class__.__name__}')
            ax1.set_ylim(bottom=0, top=1)

        ymax = 6000000  # (round(max([max(self.statistics[user.name][3]) + 1 for user in self.cluster.users])) // 10 + 1) * 10
        yticks = [1000000*i for i in range(ymax // 1000000 + 1)]

        for user in self.cluster.users:
            ax2.plot(x, self.statistics[user.name][3], self.user_color[user.name])
            ax2.set_ylim(bottom=0, top=ymax, auto=False)
            ax2.set_yticks(yticks)
        plt.show()

    def move_time(self):
        if self.cluster.queue.qsize() == 0:
            if len(self.cluster.tasks) > 0:
                min_delay = min([t.delay for t in self.cluster.tasks])
                time_difference = min_delay - self.time
                if time_difference <= 0:
                    raise Exception('Weird')
                self.time = min_delay
                return time_difference
            return 0

        peep = self.cluster.queue.get()
        if self.time >= peep[0]:
            raise Exception('There are queued tasks that finished already')
        # print(f'Moving time from {self.time} to {peep[0]}')
        time_difference = peep[0] - self.time
        for item in self.cluster.queue.queue:
            resource, task = item[2]
            self.cluster.users_share[task.user.name][1] += task.memory * time_difference
            self.cluster.users_share[task.user.name][2] += task.cores * time_difference
        self.time = peep[0]
        self.cluster.queue.put(peep)

        while not self.cluster.queue.qsize() == 0:
            item = self.cluster.queue.get()
            if self.time < item[0]:
                self.cluster.queue.put(item)
                return time_difference
            resource, task = item[2]
            resource.release_task(task)
        return time_difference

    def move_time_append_statistics(self):
        time_difference = self.move_time()
        self.append_users_statistics(time_difference)

    def max_memory_cores_available(self):
        return max([r.memory for r in self.cluster.resources]), max([r.cores for r in self.cluster.resources])

    def min_memory_cores_required(self):
        runnable_tasks = [t for t in self.cluster.tasks if t.delay <= self.time]
        if len(runnable_tasks) == 0:
            return math.inf, math.inf
        return min([t.memory for t in runnable_tasks]), min([t.cores for t in runnable_tasks])


class NetBatch(Scheduler):
    def assign_next(self):
        for user in self.cluster.get_prioritized_users():
            for task in self.cluster.get_user_tasks(user, self.time):
                for resource in self.cluster.resources:
                    if resource.verify_task(task, self.time):
                        return resource, task
        return None, None


class PenaltyAwareScheduler(Scheduler):
    @staticmethod
    def get_score(task):
        return task.penalty / task.duration

    @abstractmethod
    def assign_next(self):
        pass


class HPCBoost(PenaltyAwareScheduler):
    def assign_next(self):
        for user in self.cluster.get_prioritized_users():
            for task in sorted(self.cluster.get_user_tasks(user, self.time), key=lambda t: self.get_score(t), reverse=True):
                for resource in self.cluster.resources:
                    if resource.verify_task(task, self.time):
                        return resource, task
        return None, None


class PenaltyPerTime(PenaltyAwareScheduler):
    def assign_next(self):
        for task in sorted([task for task in self.cluster.tasks if task.delay <= self.time], key=lambda t: self.get_score(t), reverse=True):
            for resource in self.cluster.resources:
                if resource.verify_task(task, self.time):
                    return resource, task
        return None, None


class RandomScheduler(Scheduler):
    def assign_next(self):
        for _ in range(len(self.cluster.tasks)):
            task = random.choice(self.cluster.tasks)
            for _ in range(len(self.cluster.resources)):
                resource = random.choice(self.cluster.resources)
                if resource.verify_task(task, self.time):
                    return resource, task
        return None, None


class ExhaustiveRandomScheduler(Scheduler):
    def assign_next(self):
        tasks_perm = self.pick_random_perm(self.cluster.tasks)
        resources_perm = self.pick_random_perm(self.cluster.resources)
        for task in tasks_perm:
            for resource in resources_perm:
                if resource.verify_task(task, self.time):
                    return resource, task
        return None, None

    @staticmethod
    def pick_random_perm(options):
        all_task_perm = list(itertools.permutations(options))
        k = rint(len(all_task_perm) - 1)
        chosen_perm = all_task_perm[k]
        return chosen_perm


def run_scheduler(r, t, u, scheduler, n):
    for i in range(n):
        c = Cluster(r, t, u)
        print(f'{scheduler.__name__}')
        scheduler(c).run()


def rint(x):
    return random.randint(0, x)


def ruser(users):
    return random.choices(users, [user.share + 0.5*random.random() for user in users])[0]


def normalize_users_share(users):
    print([u.share for u in users])
    total_share = sum([user.share for user in users])
    for user in users:
        user.share = user.share / total_share
    print([u.share for u in users])


def rresource():
    memory, cores = random.choices(RESOURCES_MEMORY_AND_CORES)[0]
    return Resource(memory, cores)


def rmemory():
    return random.choices(TASK_MEMORIES_DISTRIBUTION[0], TASK_MEMORIES_DISTRIBUTION[1])[0]


def rdelay(timespan):
    return np.floor(timespan / 10 * (rint(10) + rint(10) / 10))


def rpenalty():
    return random.choices(TASK_PENALTY_DISTRIBUTION[0], TASK_PENALTY_DISTRIBUTION[1])[0]


def rduration():
    return random.choices(TASK_DURATION_DISTRIBUTION[0], TASK_DURATION_DISTRIBUTION[1])[0]


def rcores():
    # return 1
    return rint(C) + 1


def rtask(users, timespan):
    return Task(rmemory(), rcores(), rpenalty(), rduration(), rdelay(timespan), ruser(users))


def estimate_run_time():
    average_duration = np.dot(TASK_DURATION_DISTRIBUTION[0], TASK_DURATION_DISTRIBUTION[1])
    average_task_memory = np.dot(TASK_MEMORIES_DISTRIBUTION[0], TASK_MEMORIES_DISTRIBUTION[1])
    average_resource_memory = np.average([r[0] for r in RESOURCES_MEMORY_AND_CORES])
    memory_bound_time = T * average_duration * average_task_memory / (average_resource_memory * R)
    average_task_cores = C / 2 + 1
    average_resource_cores = np.average([r[1] for r in RESOURCES_MEMORY_AND_CORES])
    cores_bound_time = T * average_duration * average_task_cores / (average_resource_cores * R)
    print(f'{memory_bound_time=}, {cores_bound_time=}')
    return max(memory_bound_time,cores_bound_time) * 0.6


def run_test(n):
    for i in range(n):
        random.seed(i+8192)

        u0 = [User(f'User{j}', rint(S)) for j in range(U)]
        normalize_users_share(u0)
        timespan = estimate_run_time()

        r0 = [rresource() for _ in range(R)]
        t0 = [rtask(u0, timespan) for _ in range(T)]

        run_scheduler(r0, t0, u0, NetBatch, 1)
        run_scheduler(r0, t0, u0, HPCBoost, 1)
        run_scheduler(r0, t0, u0, PenaltyPerTime, 1)


if __name__ == '__main__':
    run_test(3)


