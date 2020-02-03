# Final Project of Mobile Robot of Computer Science Master's course 

Our final project desciption:

In this project, the robot needs to move to all predefined positions to localize plants in the scenario. To move to all positions only once and follow the shortest path, we ended up with the famous Traveling Salesman Problem, an NP-Hard problem. To overcome it, we relax our restriction and developed a path planning heuristic based on a shortest path algorithm, graph theory and robot behavior methods done on our previous projects. To have a better estimation of robot localization, we implemented the particle filter. To localize plants, the robot needs first to detect and classifies them, we handle this by applying deep learning to detect objects and measure their distance using robot's sensors. As a result, our project is capable of visualizing the environment, follow a path, detect and localize objects.

Team:
Tiago Trocoli,
Jing Yang,
José Nascimento,
Camila Moura
