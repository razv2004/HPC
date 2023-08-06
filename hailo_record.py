from datetime import datetime


class HailoRecord:
    def __init__(self, record):
        for x in record:
            print(x)
        self.id = record[0]
        self.completion_date = self.resolve_datetime(record[1])
        self.group = record[2]
        self.user = record[3]
        self.job_id = record[4]
        self.tool = record[5]
        self.host = record[6]
        self.duration = record[7]
        self.queue_time = record[8]
        self.status = record[9]

        self.cores = None
        self.memory = None
        self.license = []
        self.limit = []
        self.linux64 = False
        for column in range(10, 15):
            self.resolve_resource(record[column])
        if self.cores is None:
            print(f'No cores for {self}')
        if self.memory is None:
            print(f'No memory for {self}')

    def __str__(self):
        return f"[{self.id},{self.completion_date},{self.group},{self.user},{self.job_id},{self.tool},{self.host},{self.duration},{self.queue_time}," \
               f"{self.status},{self.cores},{self.memory},{self.license},{self.limit},{self.linux64}]"

    def resolve_resource(self, resource):
        if len(resource) == 0:
            return
        if resource.startswith('CORE'):
            if self.cores is not None:
                raise Exception('Ambiguous cores')
            self.cores = int(resource[6:])
        elif resource.startswith('RAM'):
            if self.memory is not None:
                raise Exception('Ambiguous memory')
            self.memory = self.resolve_memory(resource)
        elif resource.startswith('License:'):
            self.license.append(resource[8:])
        elif resource.startswith('Limit:'):
            self.limit.append(resource[6:])
        elif resource == 'linux64':
            self.linux64 = True
        else:
            raise Exception(f'Unknown {resource=}')

    def resolve_memory(self, resource):
        memory_string = resource[4:]
        if memory_string.endswith('MB'):
            return int(memory_string[:-2]) // 1024
        if memory_string.endswith('GB'):
            return int(memory_string[:-2])
        return int(memory_string)

    def resolve_datetime(self, datetime_string):
        print(datetime_string)
        return datetime.strptime(datetime_string, "%d/%m/%Y %H:%M:%S")
