class Partition:
    def __init__(self, dim, exact_sum, limit=1):

        if exact_sum < dim * limit:
            raise ValueError("Error: sum lower than limits in partition")

        self.limit = limit
        self.dim = dim

        self.p_sum = exact_sum - dim * limit
        self.p = [0] * dim
        self.p[-1] = self.p_sum

        self.result = [0] * dim
        self.first_get_call = True
        self.a = 1
        self.s = self.p_sum

    def get(self):
        if self.first_get_call:
            self.first_get_call = False
            for i in range(len(self.result)):
                self.result[i] = self.p[i] + self.limit
            return self.result

        while self.p[0] != self.p_sum:
            if self.s == 0:
                self.a -= 1
                self.s = self.p[self.a]
            elif self.p[self.a] != self.s:
                self.a += 1
            else:
                self.p[self.a - 1] += 1
                self.s -= 1
                for i in range(self.a, len(self.p) - 1):
                    self.p[i] = 0
                self.p[-1] = self.s

                for i in range(len(self.result)):
                    self.result[i] = self.p[i] + self.limit
                return self.result

        print("Warning: returning last partition again")
        for i in range(len(self.result)):
            self.result[i] = self.p[i] + self.limit
        return self.result

    def finished(self):
        return self.p[0] == self.p_sum and not self.first_get_call

    def get_all_partitions(self):
        partitions = []
        while not self.finished():
            partitions.append(self.get().copy())
        return partitions
