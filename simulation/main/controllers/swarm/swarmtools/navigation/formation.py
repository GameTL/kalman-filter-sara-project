import math
import json


class FormationMaster:
    def __init__(self, robot, current_coords: dict, object_coords: tuple, radius=0.3, fineness=2, verbose=False):
        """ 
        current_coords: assumed that data are in this order [('TurtleBot3Burger_1', [1.3787268144635236, -0.38032573186548774, 0.0]), ('TurtleBot3Burger_2', [0.07806162991058642, 0.8007404816156147, 0.0])]
        current_coords: assumed that data are in this order [robot1,robot2,robot3] -> [5,6]
        """
        self.robot = robot
        # Retrieve parameters
        self.current_coords_dict = current_coords
        self.current_coords_tuple = list(current_coords.items())
        self.member_count = len(self.current_coords_tuple)
        self.object_coords = object_coords
        self.radius = radius
        self.fineness = fineness
        self.verbose = verbose

        # Initialize calculation parameters
        self.target_coords = []
        # Schema:
        # {
        #   "robot1": {
        #       0: (x1, y1),
        #       1: (x2, y2),
        #   },
        #   "robot2": {
        #       0: (x1, y1),
        #       1: (x2, y2),
        #       2: (x3, y3),
        #   },
        # }
        self.paths = {}
        # Schema:
        # {
        #   0: [(x1, y1), (x2, y2)],
        #   1: [(x1, y1), (x2, y2), (x3, y3)],
        # }
        self.conflict_map = {}

    def calculate_target_coords(self):
        # Calculate target coords based on radius around the object coordinates
        internal_angle = 2 * math.pi / self.member_count
        for i in range(self.member_count):
            # x' = x + rcos(theta * i) 
            x = self.object_coords[0] + (self.radius * math.cos(internal_angle * i))
            # y' = y + rsin(theta * i)
            y = self.object_coords[1] + (self.radius * math.sin(internal_angle * i))

            self.target_coords.append((self.current_coords_tuple[i][0], (x, y)))

        for i, target_coord in enumerate(self.target_coords):
            self.target_coords[i] = (target_coord[0], (round(target_coord[1][0], self.fineness - 1), round(target_coord[1][1], self.fineness - 1)))
        # print(f'{self.target_coords=}')

    def calculate_distance(self, current, end):
        return round(math.sqrt((current[0] - end[0]) ** 2 + (current[1] - end[1]) ** 2), self.fineness)

    def plan_paths(self):
        print("# ===== Calculating paths ===== #")
        for (robot_id, target) in self.target_coords:
            # print(f"Assigning {robot_id}'s path: {self.current_coords_dict[robot_id][0:2]} -> {target}") 
            print(f"[path_planning]({self.robot.getName()}) Assigning {robot_id} {self.current_coords_dict[robot_id][0:2]} -> {target}")
            path = self.path_planning_algorithm(self.current_coords_dict[robot_id][0:2], target, verbose=self.verbose)
            self.paths[robot_id] = path

    def path_planning_algorithm(self, start: tuple, end: tuple, verbose=False):
        # print(f'{start=} {end=}')
        current_coordinates = start
        step = 0
        path = {}
        correct_x = False
        correct_y = False

        while True:
            difference_x = end[0] - current_coordinates[0]
            difference_y = end[1] - current_coordinates[1]
            if verbose: print(f"{difference_x=} {difference_y=}")

            movement_options = []
            # Get closer by x-axis
            if difference_x > 0:
                # Append x + 0.01
                movement_options.append((round(current_coordinates[0] + 0.01, self.fineness), current_coordinates[1]))
            elif difference_x < 0:
                # Append x - 0.01
                movement_options.append((round(current_coordinates[0] - 0.01, self.fineness), current_coordinates[1]))
            else:
                # x already at correct position
                correct_x = True
            
            # Get closer by y-axis
            if difference_y > 0:
                # Append y + 0.01
                movement_options.append((current_coordinates[0], round(current_coordinates[1] + 0.01, self.fineness)))
            elif difference_y < 0:
                # Append y - 0.01
                movement_options.append((current_coordinates[0], round(current_coordinates[1] - 0.01, self.fineness)))
            else:
                # y already at correct position
                correct_y = True

            if verbose: print(f"Movement options before pruning: {movement_options}")

            if not correct_x or not correct_y:
                # Remove conflicting movement options if step has already been initialized
                if step in self.conflict_map:
                    movement_options = [option for option in movement_options if option not in self.conflict_map[step]]
                if verbose: print(f"Movement option(s) after pruning: {movement_options}")
                
                if verbose: print(f"Correct x; {correct_x}, Correct y; {correct_y}")
                
                if len(movement_options) == 0:
                    # Stay in place if no movement options
                    path[step] = current_coordinates
                else:
                    # 2 possibilities; 
                    # if one movement option -> choose the remaining one
                    # if 2 movement options -> choose the first one
                    path[step] = movement_options[0]
                current_coordinates = path[step]
                if verbose: print(current_coordinates)
                
                if step in self.conflict_map:
                    # Key exists
                    self.conflict_map[step].append(current_coordinates)
                else:
                    # Key doesn't exist yet
                    self.conflict_map[step] = [current_coordinates]
                
                if verbose: print(self.conflict_map)

                step += 1
            
            if correct_x and correct_y:
                return path


if __name__ == "__main__":
    #! Example current positions (coordinates of the members)
    current_positions = {'TurtleBot3Burger_1': [1.7, -1.0, 0.2], 'TurtleBot3Burger_2': [-2.0, -1.4, 0.2], 'TurtleBot3Burger_3': [-0.3, 1.6, 0.5]}
    name = 'TurtleBot3Burger_2'

    formation_master = FormationMaster(current_positions, (0.75, -0.25), radius=0.3, verbose=False)
    formation_master.calculate_target_coords()
    formation_master.plan_paths()

    # print("Final Paths:")
    # for name, path in formation_master.paths.items():
    #     print(name, path)

    paths = json.dumps(formation_master.paths)
    loading = json.loads(paths)
    # print(loading)
    print(name, loading[name])
