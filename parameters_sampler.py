import random
import numpy as np
from resource import Resource
from task import Task

U = 3  # number of users --> should be tens ~80
R = 50  # number of resources
T = 20000  # number of tasks


RESOURCES_MEMORY_AND_CORES_DISTRIBUTION = [[[1536, 32], [1024, 32], [512, 32], [256, 24], [96, 16], [72, 8]], [0.1, 0.1, 0.3, 0.3, 0.1, 0.1]]
TASK_PENALTY_DISTRIBUTION = [[10, 20], [0.5, 0.5]]
TASK_DURATION_MEMORIES_CORES_DISTRIBUTION = [[[5,1,1],[60,8,4],[300,32,8],[14400,128,16],[43200,512,24],[86400,1024,32],[60,8,2],[60,2,2],[5,2,2],[5,1,2],[60,8,6],[60,16,6],[300,32,6],[300,16,6],[3600,32,8],[3600,64,8],[3600,96,12],[3600,64,12],[14400,128,12],[14400,96,12],[86400,1024,24],[86400,512,24],[43200,1024,32],[43200,512,32],[60,2,1],[5,2,1],[60,16,4],[60,2,4],[300,16,4],[5,2,4],[3600,32,6],[60,16,8],[300,16,8],[300,64,8],[300,64,12],[3600,96,16],[14400,256,16],[14400,96,16],[14400,256,24],[43200,256,16],[43200,1024,24],[43200,256,24],[86400,512,32],[60,8,1],[60,1,1],[5,8,4],[300,8,4],[300,32,4],[5,1,4],[3600,16,6],[3600,64,6],[60,8,8],[60,32,8],[300,32,12],[300,96,12],[3600,128,16],[3600,64,16],[14400,512,24],[14400,128,24],[43200,512,16],[43200,128,16]],[0.0637,0.0451,0.0403,0.0403,0.0403,0.0403,0.0319,0.0319,0.0319,0.0319,0.0225,0.0225,0.0225,0.0225,0.0202,0.0202,0.0202,0.0202,0.0202,0.0202,0.0202,0.0202,0.0202,0.0202,0.0159,0.0159,0.0113,0.0113,0.0113,0.0113,0.0113,0.0101,0.0101,0.0101,0.0101,0.0101,0.0101,0.0101,0.0101,0.0101,0.0101,0.0101,0.0101,0.0056,0.0040,0.0028,0.0028,0.0028,0.0028,0.0028,0.0028,0.0025,0.0025,0.0025,0.0025,0.0025,0.0025,0.0025,0.0025,0.0025,0.0025]]

HOUR_LOAD = [[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22], [200, 50, 10, 10, 10, 50, 10, 10, 25, 50, 10, 50]]


def rint(x):
    return random.randint(0, x)


def ruser(users):
    return random.choices(users, [user.share + 0.5*random.random() for user in users])[0]


def rresource():
    memory, cores = random.choices(RESOURCES_MEMORY_AND_CORES_DISTRIBUTION[0], RESOURCES_MEMORY_AND_CORES_DISTRIBUTION[1])[0]
    return Resource(memory, cores)


def rduration_memory_cores():
    return random.choices(TASK_DURATION_MEMORIES_CORES_DISTRIBUTION[0], TASK_DURATION_MEMORIES_CORES_DISTRIBUTION[1])[0]


def rdelay(timespan):
    return np.floor(timespan / 10 * (rint(10) + rint(10) / 10))


def rpenalty():
    return random.choices(TASK_PENALTY_DISTRIBUTION[0], TASK_PENALTY_DISTRIBUTION[1])[0]


def rtask(users, timespan):
    return Task(rduration_memory_cores(), rpenalty(), rdelay(timespan), ruser(users))


def estimate_run_time():
    task_average_memory_duration = 0
    task_average_cores_duration = 0
    for i in range(len(TASK_DURATION_MEMORIES_CORES_DISTRIBUTION[0])):
        t = TASK_DURATION_MEMORIES_CORES_DISTRIBUTION[0][i]
        p = TASK_DURATION_MEMORIES_CORES_DISTRIBUTION[1][i]
        task_average_memory_duration += t[0] * t[1] * p
        task_average_cores_duration += t[0] * t[2] * p

    average_resource_memory = np.dot([r[0] for r in RESOURCES_MEMORY_AND_CORES_DISTRIBUTION[0]], RESOURCES_MEMORY_AND_CORES_DISTRIBUTION[1])
    memory_bound_time = T * task_average_memory_duration / (average_resource_memory * R)
    average_resource_cores = np.dot([r[1] for r in RESOURCES_MEMORY_AND_CORES_DISTRIBUTION[0]], RESOURCES_MEMORY_AND_CORES_DISTRIBUTION[1])
    cores_bound_time = T * task_average_cores_duration / (average_resource_cores * R)
    return max(memory_bound_time, cores_bound_time) * 0.7, memory_bound_time > cores_bound_time


def get_resources():
    return [rresource() for _ in range(R)]


def get_tasks(users, timespan):
    return [rtask(users, timespan) for _ in range(T)]
