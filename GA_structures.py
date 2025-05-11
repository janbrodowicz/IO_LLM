from typing import List
import os
import pandas as pd
import numpy as np
from genetic_library import Element
from random import choice, random
from enum import Enum


class Teacher:
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return f"{self.name}"

    def __eq__(self, teacher):
        if isinstance(teacher, Teacher):
            return self.name == teacher.name
        return False
    
    def __hash__(self):
        return hash(self.name)

    def get_name(self) -> str:
        return self.name
    
class Classroom:
    def __init__(self, number: int):
        self.number = number

    def __str__(self):
        return f"{self.number}"
    
    def __eq__(self, classroom):
        if isinstance(classroom, Classroom):
            return self.number == classroom.number
        return False
    
    def __hash__(self):
        return hash(self.number)

    def get_classroom(self) -> int:
        return self.number
    
class Subject:
    def __init__(self, name: str, hours: int, teachers: List[Teacher], classrooms: List[Classroom]):
        self.hours = hours
        self.hours_left = hours
        self.name = name
        self.teachers = teachers
        self.classrooms = classrooms

    def __str__(self):
        return f"{self.name}"
    
    def __eq__(self, other):
        return isinstance(other, Subject) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def get_subject_name(self) -> str:
        return self.name
    
    def get_hours_left(self) -> int:
        return self.hours_left
    
    def add_subject_to_schedule(self) -> bool:
        if self.hours_left > 0:
            self.hours_left -= 1
            return True
        else:
            return False
        
    def drop_subject_from_schedule(self) -> bool:
        if self.hours_left < self.hours:
            self.hours_left += 1
            return True
        else:
            return False
        
    def is_teacher_valid(self, teacher: Teacher) -> bool:
        if teacher in self.teachers:
            return True
        else:
            return False
        
    def is_classroom_valid(self, classroom: Classroom) -> bool:
        return True if not self.classrooms else classroom in self.classrooms

class Group:
    def __init__(self, name: str, subjects: List[Subject]):
        self.name = name
        self.subjects = subjects

    def get_name(self) -> str:
        return self.name
    
    def get_subject(self, name: str):
        for el in self.subjects:
            if el.name == name:
                return el
        return None

class Year:
    def __init__(self, name: str, dir_path: str):
        self.name = name
        self.groups: List[Group] = []
        self.load_year(dir_path)

    def load_year(self, dir_path: str):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            subject_list = []
            group_name = filename[:-5]
            df = pd.read_excel(file_path)
            for index, row in df.iterrows():
                subject_name = row['Nazwa']
                subject_hours = row['Liczba godzin']
                subject_teachers = row['Prowadzący'].split(', ')
                subject_classrooms = row['Sala']

                if isinstance(subject_classrooms, str):
                    subject_classrooms = row['Sala'].split(', ')
                else:
                    subject_classrooms = []

                teachers_list = [Teacher(teach) for teach in subject_teachers]
                classrooms_list = [Classroom(classroom) for classroom in subject_classrooms]
                subject_list.append(Subject(subject_name, int(subject_hours), teachers_list, classrooms_list))

            self.groups.append(Group(group_name, subject_list))
                    
    def get_name(self) -> str:
        return self.name
    
    def get_group(self, group_name: str) -> Group:
        for el in self.groups:
            if el.name == group_name:
                return el
        return None
    
    def get_number_of_groups(self) -> int:
        return len(self.groups)

class ScheduleField:
    def __init__(self, subject: Subject, teacher: Teacher, classroom: Classroom):
        self.subject = subject
        self.teacher = teacher
        self.classroom = classroom

    def get_subject(self) -> Subject:
        return self.subject
    
    def get_teacher(self) -> Teacher:
        return self.teacher
    
    def get_classroom(self) -> Classroom:
        return self.classroom
    
    def add_field_to_schedule(self) -> bool:
        return self.subject.add_subject_to_schedule() 

    def drop_field_from_schedule(self) -> bool:
        return self.subject.drop_subject_from_schedule()

class Schedule(Element):
    def __init__(self, year: Year, lessons_per_day: int, first_population: bool):
        self.year = year
        self.number_of_lessons_per_day = lessons_per_day
        self.schedule = np.zeros((len(year.groups), 5, self.number_of_lessons_per_day), dtype=object)
        if first_population:
            self._initialize_randomly()
        super().__init__()

    def _initialize_randomly(self):
        all_classrooms = set()
        for group in self.year.groups:
            for subject in group.subjects:
                all_classrooms.update(subject.classrooms)
        all_classrooms = list(all_classrooms)

        for group_idx, group in enumerate(self.year.groups):
            for day in range(5):
                for hour in range(self.schedule.shape[2]):
                    subject = choice(group.subjects)
                    teacher = choice(subject.teachers)
                    classroom = choice(subject.classrooms) if subject.classrooms else choice(all_classrooms)
                    field = ScheduleField(subject, teacher, classroom)
                    self.schedule[group_idx][day][hour] = field if field.add_field_to_schedule() else None

    def __str__(self):
        return str(self.schedule)
    
    def _perform_mutation(self):

        all_classrooms = set()
        for group in self.year.groups:
            for subject in group.subjects:
                all_classrooms.update(subject.classrooms)
        all_classrooms = list(all_classrooms)

        group_idx = np.random.randint(self.schedule.shape[0])
        day = np.random.randint(5)
        hour = np.random.randint(self.schedule.shape[2])

        prev_field = self.schedule[group_idx][day][hour]
        if isinstance(prev_field, ScheduleField):
            prev_field.drop_field_from_schedule()

        # Randomly pick a new field (subject, teacher, classroom)
        group = self.year.groups[group_idx]
        subject = choice(group.subjects)
        teacher = choice(subject.teachers)
        classroom = choice(subject.classrooms) if subject.classrooms else choice(all_classrooms)

        field = ScheduleField(subject, teacher, classroom)
        self.schedule[group_idx][day][hour] = field if field.add_field_to_schedule() else None
    
    def evaluate_function(self):
        penalty = 0
        subject_usage = [dict() for _ in self.year.groups]

        for day in range(5):
            for hour in range(self.schedule.shape[2]):
                teacher_usage = set()
                classroom_usage = set()

                for group_idx in range(self.schedule.shape[0]):
                    field = self.schedule[group_idx][day][hour]
                    if not isinstance(field, ScheduleField):
                        continue  # Skip empty slots

                    teacher = field.get_teacher()
                    classroom = field.get_classroom()
                    subject = field.get_subject()

                    # Check for duplicate teacher
                    if teacher in teacher_usage:
                        penalty += 10
                    else:
                        teacher_usage.add(teacher)

                    # Check for duplicate classroom
                    if classroom in classroom_usage:
                        penalty += 5
                    else:
                        classroom_usage.add(classroom)

                    # Invalid subject-teacher or classroom
                    if not subject.is_teacher_valid(teacher):
                        penalty += 20
                    if not subject.is_classroom_valid(classroom):
                        penalty += 10

                    # Count subject usage
                    if subject not in subject_usage[group_idx]:
                        subject_usage[group_idx][subject] = 0
                    subject_usage[group_idx][subject] += 1
                
        for group_idx, group in enumerate(self.year.groups):
            for subject in group.subjects:
                scheduled = subject_usage[group_idx].get(subject, 0)
                if scheduled != subject.hours:
                    penalty += abs(scheduled - subject.hours) * 15

        self.fitness = penalty
        return self.fitness
    
    def crossover(self, other):
        child = Schedule(self.year, self.number_of_lessons_per_day, first_population=False)

        for group_idx in range(self.schedule.shape[0]):
            for day in range(5):
                if random() < 0.5:
                    child.schedule[group_idx][day] = self.schedule[group_idx][day].copy()
                else:
                    child.schedule[group_idx][day] = other.schedule[group_idx][day].copy()
        child.fitness = child.evaluate_function()

        return child
    
class GeneticAlgorithm:

    class SelectionModel(Enum):
        ELITE = 1
        ROULETTE = 2
        TOURNAMENT = 3

    def __init__(self, directory: str, population_size: int, lessons_per_day: int, selection_model: SelectionModel = SelectionModel.ELITE, mutation_probability: float = 0.1):
        self.year = Year("1", directory)
        self.mutation_probability = mutation_probability
        self.selection_strategy = {
            self.SelectionModel.ELITE: self.elite_selection_model,
            self.SelectionModel.ROULETTE: self.roulette_selection_model,
            self.SelectionModel.TOURNAMENT: self.tournament_selection_model
        }[selection_model]
        self.generation_count = 0
        self.population_size = population_size
        self.lessons_per_day = lessons_per_day

    def run(self):
        population = self.first_population_generator()
        population.sort(key=lambda x: x.fitness)
        while True:
            selected = self.selection_strategy(population)
            new_population = selected.copy()
            while len(new_population) != self.population_size:
                child = choice(population).crossover(choice(population))
                if random() <= self.mutation_probability:
                    child.mutation()
                new_population.append(child)

            population = new_population
            the_best_match = min(population, key=lambda x: x.fitness)
            print(f"Generation {self.generation_count} : Best match fitness {the_best_match.fitness}")
            if self.stop_condition(the_best_match.fitness):
                print(f"Finished after {self.generation_count} generations with fitness {the_best_match.fitness}")
                break
        self.export_schedule_to_excel(the_best_match, "Wyniki/final_schedule.xlsx")

    def first_population_generator(self) -> List[Schedule]:
        return [Schedule(self.year, self.lessons_per_day, first_population=True) for _ in range(self.population_size)]
    
    def stop_condition(self, fitness: int) -> bool:
        self.generation_count += 1
        return fitness == 0 or self.generation_count >= 1000

    def elite_selection_model(self, generation: List[Schedule]) -> List[Schedule]:
        max_selected = int(len(generation) / 10)
        sorted_by_assess = sorted(generation, key=lambda x: x.fitness)
        return sorted_by_assess[:max_selected]
    
    def roulette_selection_model(self, generation: List[Schedule]) -> List[Schedule]:
        total_fitness = sum(1 / (1 + s.fitness) for s in generation)
        probs = [(1 / (1 + s.fitness)) / total_fitness for s in generation]
        selected = np.random.choice(generation, size=len(generation)//10, p=probs, replace=False)
        return list(selected)

    def tournament_selection_model(self, generation: List[Schedule]) -> List[Schedule]:
        selected = []
        for _ in range(len(generation)//10):
            contenders = [choice(generation) for _ in range(5)]
            selected.append(min(contenders, key=lambda x: x.fitness))
        return selected
    
    def export_schedule_to_excel(self, schedule: Schedule, filename: str):
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        num_lessons = schedule.schedule.shape[2]
        
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            for group_idx, group in enumerate(schedule.year.groups):
                data = []

                for hour in range(num_lessons):
                    row = []
                    for day_idx, day_name in enumerate(days):
                        field = schedule.schedule[group_idx][day_idx][hour]
                        if isinstance(field, ScheduleField):
                            subj = field.get_subject().get_subject_name()
                            teach = field.get_teacher().get_name()
                            room = field.get_classroom().get_classroom()
                            row.append(f"{subj}\n{teach}\nRoom {room}")
                        else:
                            row.append("")
                    data.append(row)

                df = pd.DataFrame(data, columns=days, index=[f"Lesson {i+1}" for i in range(num_lessons)])
                df.to_excel(writer, sheet_name=group.get_name())