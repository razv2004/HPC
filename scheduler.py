from abc import abstractmethod
import math
import numpy as np
from matplotlib import pyplot as plt

COLORS = ['r', 'g', 'b', 'c', 'm', 'y']


class Scheduler:
    def __init__(self, cluster):
        self.cluster = cluster
        self.statistics = {}  # user --> statistics
        for u in self.cluster.users:
            self.statistics[u.name] = [[], [], [], []]
        self.checkpoints = []

        self.user_color = {}
        for i in range(len(self.cluster.users)):
            self.user_color[self.cluster.users[i].name] = COLORS[i % len(COLORS)]

    @abstractmethod
    def assign_next(self):
        pass

    def run(self):
        print(f'{self.__class__.__name__}')
        while not self.cluster.is_done():
            # print(f'{self.cluster.time=}, {self.cluster.left_tasks=}, {self.cluster.queue.qsize()=}')
            if self.cluster.left_tasks == 0 or self.are_memory_and_cores_insufficient():
                self.move_time_append_statistics()
                continue
            resource, task = self.assign_next()
            if resource is None or task is None:
                self.move_time_append_statistics()
                continue
            self.cluster.fair_share_monitor[task.user.name][3] += task.get_task_score(self.cluster.time)
            self.cluster.start_task(resource, task)
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
            user_share = self.cluster.fair_share_monitor[user.name]
            self.statistics[user.name][0].append(user_share[0])
            self.statistics[user.name][1].append(user_share[1] / (self.cluster.available_memory_per_hour * max(self.cluster.time, 1)))
            self.statistics[user.name][2].append(user_share[2] / (self.cluster.available_cores_per_hour * max(self.cluster.time, 1)))
            self.statistics[user.name][3].append(user_share[3])
        self.checkpoints.append(time_difference)

    def print_statistics(self):
        x = np.cumsum(self.checkpoints)

        # figure, (ax1, ax2, ax3) = plt.subplots(3)

        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        fig.suptitle(f'{self.__class__.__name__}', fontsize=16)
        fig.tight_layout(pad=2.0)

        ax1.title.set_text('Memory')
        for user in self.cluster.users:
            ax1.plot(x, self.statistics[user.name][0], self.user_color[user.name] + '--')
            ax1.plot(x, self.statistics[user.name][1], self.user_color[user.name] + '-')
            # ax1.set_title(f'{self.__class__.__name__},{min(self.statistics[user.name][3])},{max(self.statistics[user.name][3])}')

        ax1.plot(x, np.sum([self.statistics[user.name][1] for user in self.cluster.users], 0), 'k-')
        ax1.set_ylim(bottom=0, top=1)

        ax2.title.set_text('Cores')
        for user in self.cluster.users:
            ax2.plot(x, self.statistics[user.name][0], self.user_color[user.name] + '--')
            ax2.plot(x, self.statistics[user.name][2], self.user_color[user.name] + '-')
            # ax2.set_title(f'{self.__class__.__name__},{min(self.statistics[user.name][3])},{max(self.statistics[user.name][3])}')

        ax2.set_ylim(bottom=0, top=1)
        ax2.plot(x, np.sum([self.statistics[user.name][2] for user in self.cluster.users], 0), 'k-')

        yunit_raw = max([max(self.statistics[user.name][3]) + 1 for user in self.cluster.users]) // 5
        log_order = math.log(yunit_raw, 10)
        order = 10 ** int(math.floor(log_order))
        yunit = math.ceil(yunit_raw // order + 1) * order
        ymax = yunit * 5  # (round(max([max(self.statistics[user.name][3]) + 1 for user in self.cluster.users])) // 10 + 1) * 10
        yticks = [yunit * i for i in range(6)]

        ax3.title.set_text('Score')
        for user in self.cluster.users:
            ax3.plot(x, self.statistics[user.name][3], self.user_color[user.name])
            ax3.set_ylim(bottom=0, top=ymax, auto=False)
            ax3.set_yticks(yticks)
        plt.show()

    def move_time(self):
        if self.cluster.queue.qsize() == 0:
            if len(self.cluster.potential_tasks) > 0:
                min_delay = min([t.delay for t in self.cluster.potential_tasks])
                time_difference = min_delay - self.cluster.time
                self.cluster.update_time(min_delay)
                return time_difference
            return 0

        peep = self.cluster.queue.get()
        if self.cluster.time >= peep[0]:
            raise Exception('There are queued tasks that finished already')
        time_difference = peep[0] - self.cluster.time
        for item in self.cluster.queue.queue:
            resource, task = item[2]
            self.cluster.fair_share_monitor[task.user.name][1] += task.memory * time_difference
            self.cluster.fair_share_monitor[task.user.name][2] += task.cores * time_difference
        self.cluster.update_time(peep[0])
        self.cluster.queue.put(peep)

        while not self.cluster.queue.qsize() == 0:
            item = self.cluster.queue.get()
            if self.cluster.time < item[0]:
                self.cluster.queue.put(item)
                return time_difference
            resource, task = item[2]
            self.cluster.end_task(resource, task)
        return time_difference

    def move_time_append_statistics(self):
        time_difference = self.move_time()
        self.append_users_statistics(time_difference)

    def max_memory_cores_available(self):
        if len(self.cluster.current_resources) == 0:
            return -math.inf, -math.inf
        return max([r.memory for r in self.cluster.current_resources]), max([r.cores for r in self.cluster.current_resources])

    def min_memory_cores_required(self):
        if len(self.cluster.current_tasks) == 0:
            return math.inf, math.inf
        return min([t.memory for t in self.cluster.current_tasks]), min([t.cores for t in self.cluster.current_tasks])


class Backfill(Scheduler):
    def assign_next(self):
        for u in self.cluster.get_prioritized_users():
            for t in self.cluster.get_user_tasks(u):
                for r in self.cluster.current_tasks[t]:
                    return r, t
        return None, None


class PenaltyAwareScheduler(Scheduler):
    @staticmethod
    def get_score(task):
        return task.penalty / task.duration

    @abstractmethod
    def assign_next(self):
        pass


class UserBeforeTask(PenaltyAwareScheduler):
    def assign_next(self):
        for u in self.cluster.get_prioritized_users():
            for t in sorted(self.cluster.get_user_tasks(u), key=lambda t1: self.get_score(t1), reverse=True):
                for r in self.cluster.current_tasks[t]:
                    return r, t
        return None, None


class TaskWithCombinedScore(Scheduler):
    def assign_next(self):
        for t in sorted(self.cluster.current_tasks, key=lambda t1: self.get_score(t1), reverse=True):
            for r in self.cluster.current_tasks[t]:
                return r, t
        return None, None

    def get_score(self, task):
        return task.penalty * task.duration
        # user_share = self.cluster.fair_share_monitor[task.user.name]
        # cores_share = max(user_share[2], 1) / (self.cluster.available_cores_per_hour * max(self.cluster.time, 1))
        # return task.cores * task.penalty / cores_share


class FullSharing(PenaltyAwareScheduler):
    def assign_next(self):
        for t in sorted(self.cluster.current_tasks, key=lambda t1: self.get_score(t1), reverse=True):
            for r in self.cluster.current_tasks[t]:
                return r, t
        return None, None
