import numpy as np
from numpy.random import choice as np_choice
import matplotlib.pyplot as plt
import distancesData as distancesData
import citiesData as citiesData


class AntAlgorhytm(object):
    def __init__(self, distancesMatrix, antsCount, bestAntsCount, iterationsCount, feromoneLifetime, alpha=1.0,
                 beta=1.0):
        """
            distancesMatrix (2D numpy.array): Квадратна матриця відстаней. Діагональ вважається np.inf.
            antsCount (int): Кількість мурах, що запускаються за ітерацію
            n_best (int): Кількість кращих мурах, які відкладають феромон
            n_iteration (int): Кількість ітерацій
            feromoneLifetime (float): Швидкість розпаду феромону
            alpha (int or float): експонента на феромоні, вища альфа надає феромону більшої ваги. Default=1
            beta (int or float): експонента на дистанції, вища бета надає дистанції більшої ваги. Default=1
        """
        self.distancesMatrix = distancesMatrix
        self.pheromone = np.ones(self.distancesMatrix.shape) / len(distancesMatrix)
        self.all_inds = range(len(distancesMatrix))
        self.antsCount = antsCount
        self.bestAntsCount = bestAntsCount
        self.iterationsCount = iterationsCount
        self.decay = feromoneLifetime
        self.alpha = alpha
        self.beta = beta

    def Launch(self, startCity=0):
        # Пошук найкоротшого шляху
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)
        for i in range(self.iterationsCount):
            all_paths = self.GetAllPathsDistances(startCity)
            self.SpreadPheromone(all_paths, self.bestAntsCount, shortest_path=shortest_path)
            shortest_path = min(all_paths, key=lambda x: x[1])
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            self.pheromone = self.pheromone * self.decay

        return all_time_shortest_path

    def SpreadPheromone(self, all_paths, n_best, shortest_path):
        # Задання значення феромонів
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                self.pheromone[move] += 1.0 / self.distancesMatrix[move]

    def GetPathDistance(self, path):
        # Отримання довжини шляху
        total_dist = 0
        for ele in path:
            total_dist += self.distancesMatrix[ele]

        return total_dist

    def GetAllPathsDistances(self, start):
        # Отримання довжини всіх шляхів
        all_paths = []
        for i in range(self.antsCount):
            path = self.GeneratePath(start)
            all_paths.append((path, self.GetPathDistance(path)))

        return all_paths

    def GeneratePath(self, start):
        # Переміщення до наступного пункту
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distancesMatrix) - 1):
            move = self.SelectNextVertex(self.pheromone[prev], self.distancesMatrix[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start))  # повернення на початок

        return path

    def SelectNextVertex(self, pheromone, dist, visited):
        # Вибір наступного пункту переміщення
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0

        row = pheromone ** self.alpha * ((1.0 / dist) ** self.beta)

        norm_row = row / row.sum()
        move = np_choice(self.all_inds, 1, p=norm_row)[0]

        return move


if __name__ == '__main__':
    # Пошук найкоротшого маршруту
    pathfinder = AntAlgorhytm(distancesData.distances, 20, 20, 100, 1, alpha=1, beta=1)

    result = pathfinder.Launch(startCity=15)  # варіант 16
    print(f"Отриманий найкоротший шлях: {result[1]} км")

    # Виведення знайденого шляху
    path = "Шлях: \n"
    for i in result[0]:
        path += f" -> {citiesData.cities[i[0]]} \n"
    print(path)

    # Побудова графіку найкоротшого маршруту
    fig = plt.figure(figsize=(13, 13))
    plt.xticks([i + 1 for i in range(25)])
    plt.yticks([i for i in range(25)], citiesData.cities)
    plt.xlabel("Номери міст")
    plt.ylabel("Назви міст")
    plt.title("Маршрут, пройдений коміявожером")
    plt.plot([i + 1 for i in range(len(result[0]))], [i[0] for i in result[0]], ms=12, marker='x', mfc='b', mew=2,
             color='#0077b6')
    plt.grid()
    plt.show()
