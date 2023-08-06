import copy
from queue import PriorityQueue
from collections import defaultdict


def debug_me():
    print(1)


class Cluster:
    def __init__(self, resources, tasks, users):
        self.waiting_tasks = copy.deepcopy(tasks)
        self.left_tasks = len(self.waiting_tasks)

        self.potential_resources = defaultdict(set)
        self.potential_tasks = defaultdict(set)
        self.current_tasks = defaultdict(set)
        self.current_resources = defaultdict(set)

        for r in copy.deepcopy(resources):
            for t in self.waiting_tasks:
                if r.verify_task(t):
                    # self.potential_resources[r].add(t)
                    self.potential_tasks[t].add(r)

        max_memory = max([r.memory for r in resources])
        max_cores = max([r.cores for r in resources])

        impossible_tasks = [t for t in tasks if t.memory > max_memory or t.cores > max_cores]
        if len(impossible_tasks) > 0:
            print(f'{[t.__str__() for t in impossible_tasks]}')
            raise Exception('Impossible tasks were found')

        self.available_memory_per_hour = sum([r.memory for r in resources])
        self.available_cores_per_hour = sum([r.cores for r in resources])

        self.time = -1

        self.users = copy.deepcopy(users)
        self.fair_share_monitor = self.init_fair_share()

        self.queue = PriorityQueue()
        # self.current_running = {}
        # self.verify_potential()

    def init_fair_share(self):
        fair_share = {}
        total_share = sum([u.share for u in self.users])
        for u in self.users:
            fair_share[u.name] = [u.share / total_share, 0, 0, 0]  # 0s represent no core/time/penalty so far
        return fair_share

    def __str__(self):
        return f"Cluster({len(self.current_tasks)} --- {self.queue.qsize()})"

    def update_time(self, time):
        previous_time = self.time
        self.time = time

        for t in self.waiting_tasks:
            if previous_time < t.delay <= self.time:
                self.current_tasks[t] = set()
                for r in self.potential_tasks[t]:
                    if r.verify_task(t):
                        self.current_tasks[t].add(r)
                        self.current_resources[r].add(t)

        self.waiting_tasks = [t for t in self.waiting_tasks if t.delay > self.time]

    def start_task(self, resource, task):
        # self.verify_current()
        self.reserve_resource(resource, task)
        self.left_tasks -= 1
        finish_time = self.time + task.duration
        self.queue.put((finish_time, self.left_tasks, (resource, task)))

        for r in self.potential_tasks[task]:
            self.potential_resources[r].discard(task)
            self.current_resources[r].discard(task)
        del self.potential_tasks[task]
        del self.current_tasks[task]

        # self.verify_current()

    def end_task(self, resource, task):
        # self.verify_current()
        resource.release(task)
        self.current_resources[resource] = set(filter(lambda t: resource.verify_task(t), self.current_tasks))

        for t in self.current_resources[resource]:
            self.current_tasks[t].add(resource)

        # self.verify_current()

    def reserve_resource(self, resource, task):
        resource.reserve(task, self.time)

        for t in self.current_resources[resource]:
            if not resource.verify_task(t):
                self.current_tasks[t].discard(resource)

        self.current_resources[resource].discard(task)
        self.current_resources[resource] = set(filter(lambda t: resource.verify_task(t), self.current_resources[resource]))

    def get_user_tasks(self, user):
        return [t for t in self.current_tasks if t.user.name == user.name]

    def get_prioritized_users(self):
        return sorted(self.users, key=lambda u: self.fair_share_monitor[u.name][1] / self.fair_share_monitor[u.name][0])

    def is_done(self):
        return self.left_tasks == 0 and self.queue.qsize() == 0

    def verify_potential(self):
        for t in self.potential_tasks:
            if t not in self.waiting_tasks:
                debug_me()
            for r in self.potential_tasks[t]:
                if r not in self.potential_resources:
                    debug_me()
                if t not in self.potential_resources[r]:
                    debug_me()
                if not r.verify_task(t):
                    debug_me()

        for r in self.potential_resources:
            for t in self.potential_resources[r]:
                if t not in self.waiting_tasks:
                    debug_me()
                if t not in self.potential_tasks:
                    debug_me()
                if not r in self.potential_tasks[t]:
                    debug_me()
                if not r.verify_task(t):
                    debug_me()

        for t in self.waiting_tasks:
            if t not in self.potential_tasks:
                debug_me()
            if t in self.current_tasks:
                debug_me()

    def verify_current(self):
        for t in self.current_tasks:
            if t.delay > self.time:
                debug_me()
            if not t in self.potential_tasks:
                debug_me()
            if t in self.waiting_tasks:
                debug_me()
            for r in self.current_tasks[t]:
                if not r in self.potential_resources:
                    debug_me()
                if not r in self.current_resources:
                    debug_me()
                if not r in self.potential_tasks[t]:
                    debug_me()
                if not t in self.potential_resources[r]:
                    debug_me()
                if not t in self.current_resources[r]:
                    debug_me()
                if not r.verify_task(t):
                    debug_me()

        for r in self.current_resources:
            if not r in self.potential_resources:
                debug_me()
            for t in self.current_resources[r]:
                if t.delay > self.time:
                    debug_me()
                if t in self.waiting_tasks:
                    debug_me()
                if not t in self.potential_tasks:
                    debug_me()
                if not t in self.current_tasks:
                    debug_me()
                if not t in self.potential_resources[r]:
                    debug_me()
                if not r in self.potential_tasks[t]:
                    debug_me()
                if not r in self.current_tasks[t]:
                    debug_me()
                if not r.verify_task(t):
                    debug_me()

        for t in self.waiting_tasks:
            if t not in self.potential_tasks:
                debug_me()
            if t in self.current_tasks:
                debug_me()
