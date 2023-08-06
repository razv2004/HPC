import random
import time
import parameters_sampler as ps
from cluster import Cluster
from scheduler import Backfill, UserBeforeTask, TaskWithCombinedScore, FullSharing
from user import User


def run_scheduler(cluster, scheduler):
    scheduler(cluster).run()


def run_test(seeds):
    for seed in seeds:
        random.seed(seed)

        users = [User('User0', 10), User('User1', 30), User('User2', 60)]
        timespan, memory_bound = ps.estimate_run_time()
        share = [u.share for u in users]
        print(f'{seed=}, {share=}, {timespan=}, {memory_bound=}')

        resources = ps.get_resources()
        tasks = ps.get_tasks(users, timespan)

        for scheduler in [TaskWithCombinedScore, Backfill, FullSharing]:
            start = time.time()
            cluster = Cluster(resources, tasks, users)
            run_scheduler(cluster, scheduler)
            print(time.time() - start)


if __name__ == '__main__':
    run_test([i for i in range(500, 501)])


