import numpy as np
from typing import Dict, List, Any, Tuple
from enum import Enum
import random
import time

class DifficultyLevel(Enum):
    """Difficulty levels for curriculum learning"""
    EASY = "easy"
    MEDIUM = "medium" 
    HARD = "hard"
    EXPERT = "expert"
    MASTER = "master"

class ScenarioType(Enum):
    """Types of training scenarios"""
    NORMAL_TRAFFIC = "normal_traffic"
    RUSH_HOUR = "rush_hour"
    EMERGENCY_RESPONSE = "emergency_response"
    CONGESTION = "congestion"
    ACCIDENT = "accident"
    WEATHER_EVENT = "weather_event"
    SPECIAL_EVENT = "special_event"

class CurriculumStage:
    """Represents a stage in the curriculum"""
    
    def __init__(self, stage_id: int, name: str, difficulty: DifficultyLevel, 
                 scenarios: List[ScenarioType], performance_threshold: float,
                 max_duration: int = 300,  # 5 minutes default
                 description: str = ""):
        self.stage_id = stage_id
        self.name = name
        self.difficulty = difficulty
        self.scenarios = scenarios
        self.performance_threshold = performance_threshold
        self.max_duration = max_duration
        self.description = description
        self.completed = False
        self.attempts = 0
        self.best_performance = 0.0
        self.start_time = None
        
    def start_stage(self):
        """Start this curriculum stage"""
        self.start_time = time.time()
        self.attempts += 1
        print(f"🎯 Starting curriculum stage: {self.name} (Difficulty: {self.difficulty.value})")
        
    def check_completion(self, performance_metrics: Dict) -> Tuple[bool, float]:
        """Check if stage is completed based on performance"""
        performance_score = self._calculate_performance_score(performance_metrics)
        self.best_performance = max(self.best_performance, performance_score)
        
        # Check if performance threshold is met
        completed = performance_score >= self.performance_threshold
        
        # Check if max duration exceeded
        if self.start_time and time.time() - self.start_time > self.max_duration:
            completed = True
            print(f"⏰ Stage {self.name} time limit reached")
        
        if completed and not self.completed:
            self.completed = True
            print(f"✅ Stage {self.name} completed! Performance: {performance_score:.3f}")
        
        return completed, performance_score
    
    def _calculate_performance_score(self, metrics: Dict) -> float:
        """Calculate performance score for this stage"""
        base_score = 0.0
        
        if self.difficulty == DifficultyLevel.EASY:
            # Focus on basic traffic flow
            waiting_time = metrics.get('total_waiting_time', 0)
            vehicles_cleared = metrics.get('vehicles_cleared', 0)
            base_score = self._easy_scoring(waiting_time, vehicles_cleared)
            
        elif self.difficulty == DifficultyLevel.MEDIUM:
            # Include emergency response
            emergency_time = metrics.get('emergency_response_time', 30)
            base_score = self._medium_scoring(metrics, emergency_time)
            
        elif self.difficulty == DifficultyLevel.HARD:
            # Complex scenarios with multiple metrics
            base_score = self._hard_scoring(metrics)
            
        elif self.difficulty in [DifficultyLevel.EXPERT, DifficultyLevel.MASTER]:
            # Expert-level performance requirements
            base_score = self._expert_scoring(metrics)
        
        # Adjust score based on scenario type
        scenario_bonus = self._calculate_scenario_bonus(metrics)
        final_score = min(base_score + scenario_bonus, 1.0)
        
        return final_score
    
    def _easy_scoring(self, waiting_time: float, vehicles_cleared: int) -> float:
        """Scoring for easy difficulty"""
        waiting_score = 1.0 - min(waiting_time / 500.0, 1.0)
        throughput_score = min(vehicles_cleared / 20.0, 1.0)
        return (waiting_score * 0.6 + throughput_score * 0.4)
    
    def _medium_scoring(self, metrics: Dict, emergency_time: float) -> float:
        """Scoring for medium difficulty"""
        waiting_score = 1.0 - min(metrics.get('total_waiting_time', 0) / 300.0, 1.0)
        throughput_score = min(metrics.get('vehicles_cleared', 0) / 30.0, 1.0)
        emergency_score = 1.0 - min(emergency_time / 20.0, 1.0)
        
        return (waiting_score * 0.4 + throughput_score * 0.3 + emergency_score * 0.3)
    
    def _hard_scoring(self, metrics: Dict) -> float:
        """Scoring for hard difficulty"""
        waiting_score = 1.0 - min(metrics.get('total_waiting_time', 0) / 200.0, 1.0)
        throughput_score = min(metrics.get('vehicles_cleared', 0) / 40.0, 1.0)
        avg_speed_score = min(metrics.get('average_speed', 0) / 15.0, 1.0)
        emergency_score = 1.0 - min(metrics.get('emergency_response_time', 30) / 15.0, 1.0)
        
        return (waiting_score * 0.3 + throughput_score * 0.3 + 
                avg_speed_score * 0.2 + emergency_score * 0.2)
    
    def _expert_scoring(self, metrics: Dict) -> float:
        """Scoring for expert/master difficulty"""
        waiting_score = 1.0 - min(metrics.get('total_waiting_time', 0) / 150.0, 1.0)
        throughput_score = min(metrics.get('vehicles_cleared', 0) / 50.0, 1.0)
        avg_speed_score = min(metrics.get('average_speed', 0) / 18.0, 1.0)
        emergency_score = 1.0 - min(metrics.get('emergency_response_time', 30) / 10.0, 1.0)
        fuel_efficiency = metrics.get('fuel_efficiency', 0.5)  # Placeholder
        
        return (waiting_score * 0.25 + throughput_score * 0.25 + 
                avg_speed_score * 0.2 + emergency_score * 0.2 + fuel_efficiency * 0.1)
    
    def _calculate_scenario_bonus(self, metrics: Dict) -> float:
        """Calculate bonus based on scenario performance"""
        bonus = 0.0
        
        if ScenarioType.EMERGENCY_RESPONSE in self.scenarios:
            emergencies_handled = metrics.get('emergencies_handled', 0)
            bonus += min(emergencies_handled / 5.0, 0.2)
        
        if ScenarioType.CONGESTION in self.scenarios:
            congestion_cleared = metrics.get('congestion_cleared', 0)
            bonus += min(congestion_cleared / 3.0, 0.15)
        
        return bonus

class ScenarioGenerator:
    """Generates training scenarios based on curriculum stage"""
    
    def __init__(self):
        self.scenario_templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[ScenarioType, Dict]:
        """Initialize scenario templates"""
        return {
            ScenarioType.NORMAL_TRAFFIC: {
                'vehicle_density': 0.3,
                'emergency_probability': 0.05,
                'spawn_rate': 2.0,
                'max_vehicles': 15,
                'description': 'Normal traffic conditions'
            },
            ScenarioType.RUSH_HOUR: {
                'vehicle_density': 0.8,
                'emergency_probability': 0.02,
                'spawn_rate': 5.0,
                'max_vehicles': 30,
                'description': 'Rush hour heavy traffic'
            },
            ScenarioType.EMERGENCY_RESPONSE: {
                'vehicle_density': 0.4,
                'emergency_probability': 0.2,
                'spawn_rate': 3.0,
                'max_vehicles': 20,
                'emergency_types': ['ambulance', 'fire_truck', 'police'],
                'description': 'Multiple emergency vehicles'
            },
            ScenarioType.CONGESTION: {
                'vehicle_density': 0.9,
                'emergency_probability': 0.01,
                'spawn_rate': 6.0,
                'max_vehicles': 35,
                'congestion_zones': ['J1-J2', 'J3-J4'],
                'description': 'Severe traffic congestion'
            },
            ScenarioType.ACCIDENT: {
                'vehicle_density': 0.5,
                'emergency_probability': 0.3,
                'spawn_rate': 2.0,
                'max_vehicles': 18,
                'blocked_lanes': 1,
                'description': 'Traffic accident scenario'
            },
            ScenarioType.WEATHER_EVENT: {
                'vehicle_density': 0.4,
                'emergency_probability': 0.08,
                'spawn_rate': 2.5,
                'max_vehicles': 16,
                'speed_reduction': 0.7,
                'description': 'Adverse weather conditions'
            },
            ScenarioType.SPECIAL_EVENT: {
                'vehicle_density': 0.7,
                'emergency_probability': 0.1,
                'spawn_rate': 4.0,
                'max_vehicles': 25,
                'special_routes': True,
                'description': 'Special event traffic patterns'
            }
        }
    
    def generate_scenario_parameters(self, scenario_type: ScenarioType, 
                                   difficulty: DifficultyLevel) -> Dict[str, Any]:
        """Generate parameters for a specific scenario type and difficulty"""
        template = self.scenario_templates.get(scenario_type, {})
        params = template.copy()
        
        # Adjust parameters based on difficulty
        difficulty_multipliers = {
            DifficultyLevel.EASY: 0.7,
            DifficultyLevel.MEDIUM: 1.0,
            DifficultyLevel.HARD: 1.3,
            DifficultyLevel.EXPERT: 1.6,
            DifficultyLevel.MASTER: 2.0
        }
        
        multiplier = difficulty_multipliers.get(difficulty, 1.0)
        
        # Adjust key parameters
        if 'vehicle_density' in params:
            params['vehicle_density'] = min(params['vehicle_density'] * multiplier, 0.95)
        
        if 'spawn_rate' in params:
            params['spawn_rate'] *= multiplier
        
        if 'max_vehicles' in params:
            params['max_vehicles'] = int(params['max_vehicles'] * multiplier)
        
        if 'emergency_probability' in params:
            params['emergency_probability'] = min(params['emergency_probability'] * multiplier, 0.5)
        
        # Add difficulty-specific challenges
        if difficulty in [DifficultyLevel.HARD, DifficultyLevel.EXPERT, DifficultyLevel.MASTER]:
            params['dynamic_events'] = True
            params['multiple_emergencies'] = True
        
        if difficulty in [DifficultyLevel.EXPERT, DifficultyLevel.MASTER]:
            params['simultaneous_challenges'] = True
            params['performance_penalties'] = True
        
        return params
    
    def get_random_scenario(self, difficulty: DifficultyLevel) -> Tuple[ScenarioType, Dict]:
        """Get a random scenario for the given difficulty"""
        available_scenarios = list(self.scenario_templates.keys())
        scenario_type = random.choice(available_scenarios)
        parameters = self.generate_scenario_parameters(scenario_type, difficulty)
        
        return scenario_type, parameters

class CurriculumManager:
    """Manages the curriculum learning process"""
    
    def __init__(self):
        self.stages = []
        self.current_stage_index = 0
        self.scenario_generator = ScenarioGenerator()
        self.learning_history = []
        self.performance_trend = []
        
        # Initialize curriculum stages
        self._initialize_curriculum()
    
    def _initialize_curriculum(self):
        """Initialize the curriculum stages"""
        self.stages = [
            CurriculumStage(
                stage_id=1,
                name="Basic Traffic Control",
                difficulty=DifficultyLevel.EASY,
                scenarios=[ScenarioType.NORMAL_TRAFFIC],
                performance_threshold=0.7,
                max_duration=600,  # 10 minutes
                description="Learn basic traffic light control in normal conditions"
            ),
            CurriculumStage(
                stage_id=2,
                name="Emergency Response Basics",
                difficulty=DifficultyLevel.EASY,
                scenarios=[ScenarioType.NORMAL_TRAFFIC, ScenarioType.EMERGENCY_RESPONSE],
                performance_threshold=0.65,
                max_duration=600,
                description="Handle occasional emergency vehicles"
            ),
            CurriculumStage(
                stage_id=3,
                name="Moderate Traffic",
                difficulty=DifficultyLevel.MEDIUM,
                scenarios=[ScenarioType.NORMAL_TRAFFIC, ScenarioType.RUSH_HOUR],
                performance_threshold=0.7,
                max_duration=600,
                description="Manage increased traffic volume"
            ),
            CurriculumStage(
                stage_id=4,
                name="Complex Scenarios",
                difficulty=DifficultyLevel.MEDIUM,
                scenarios=[ScenarioType.EMERGENCY_RESPONSE, ScenarioType.CONGESTION],
                performance_threshold=0.65,
                max_duration=600,
                description="Handle emergencies during congestion"
            ),
            CurriculumStage(
                stage_id=5,
                name="Advanced Traffic Management",
                difficulty=DifficultyLevel.HARD,
                scenarios=[ScenarioType.RUSH_HOUR, ScenarioType.EMERGENCY_RESPONSE, ScenarioType.CONGESTION],
                performance_threshold=0.6,
                max_duration=600,
                description="Manage multiple challenging scenarios simultaneously"
            ),
            CurriculumStage(
                stage_id=6,
                name="Expert Scenarios",
                difficulty=DifficultyLevel.EXPERT,
                scenarios=[ScenarioType.ACCIDENT, ScenarioType.WEATHER_EVENT, ScenarioType.SPECIAL_EVENT],
                performance_threshold=0.55,
                max_duration=600,
                description="Handle rare but critical scenarios"
            ),
            CurriculumStage(
                stage_id=7,
                name="Master Level",
                difficulty=DifficultyLevel.MASTER,
                scenarios=list(ScenarioType),  # All scenarios
                performance_threshold=0.5,
                max_duration=600,
                description="Master all traffic control scenarios"
            )
        ]
        
        print("📚 Curriculum initialized with 7 stages")
    
    def get_current_stage(self) -> CurriculumStage:
        """Get the current curriculum stage"""
        if self.current_stage_index < len(self.stages):
            return self.stages[self.current_stage_index]
        else:
            # All stages completed, return the last stage for continued training
            return self.stages[-1]
    
    def start_training_session(self) -> Dict[str, Any]:
        """Start a new training session with current stage parameters"""
        current_stage = self.get_current_stage()
        current_stage.start_stage()
        
        # Select scenario for this session
        if current_stage.scenarios:
            scenario_type = random.choice(current_stage.scenarios)
        else:
            scenario_type = ScenarioType.NORMAL_TRAFFIC
        
        scenario_params = self.scenario_generator.generate_scenario_parameters(
            scenario_type, current_stage.difficulty
        )
        
        training_parameters = {
            'stage': current_stage.name,
            'difficulty': current_stage.difficulty.value,
            'scenario_type': scenario_type.value,
            'scenario_parameters': scenario_params,
            'performance_threshold': current_stage.performance_threshold,
            'max_duration': current_stage.max_duration,
            'stage_id': current_stage.stage_id
        }
        
        print(f"🎯 Training Session: {current_stage.name}")
        print(f"   Scenario: {scenario_type.value}")
        print(f"   Difficulty: {current_stage.difficulty.value}")
        print(f"   Target Performance: {current_stage.performance_threshold}")
        
        return training_parameters
    
    def update_progress(self, performance_metrics: Dict) -> Dict[str, Any]:
        """Update curriculum progress based on performance"""
        current_stage = self.get_current_stage()
        completed, performance_score = current_stage.check_completion(performance_metrics)
        
        # Record learning history
        history_entry = {
            'timestamp': time.time(),
            'stage_id': current_stage.stage_id,
            'stage_name': current_stage.name,
            'performance_score': performance_score,
            'metrics': performance_metrics,
            'completed': completed
        }
        self.learning_history.append(history_entry)
        self.performance_trend.append(performance_score)
        
        # Keep only recent history
        if len(self.learning_history) > 1000:
            self.learning_history.pop(0)
        if len(self.performance_trend) > 100:
            self.performance_trend.pop(0)
        
        progress_info = {
            'current_stage': current_stage.name,
            'performance_score': performance_score,
            'best_performance': current_stage.best_performance,
            'threshold': current_stage.performance_threshold,
            'completed': completed,
            'stage_completed': False,
            'curriculum_completed': False
        }
        
        # Advance to next stage if completed
        if completed and self.current_stage_index < len(self.stages) - 1:
            self.current_stage_index += 1
            progress_info['stage_completed'] = True
            print(f"🎉 Advanced to stage {self.current_stage_index + 1}: {self.get_current_stage().name}")
        
        # Check if entire curriculum is completed
        if self.current_stage_index == len(self.stages) - 1 and current_stage.completed:
            progress_info['curriculum_completed'] = True
            print("🏆 Curriculum completed! Mastered all stages!")
        
        return progress_info
    
    def get_curriculum_stats(self) -> Dict[str, Any]:
        """Get curriculum statistics"""
        completed_stages = sum(1 for stage in self.stages if stage.completed)
        total_attempts = sum(stage.attempts for stage in self.stages)
        
        if self.performance_trend:
            recent_performance = np.mean(self.performance_trend[-10:])
        else:
            recent_performance = 0.0
        
        return {
            'total_stages': len(self.stages),
            'completed_stages': completed_stages,
            'current_stage_index': self.current_stage_index,
            'current_stage': self.get_current_stage().name,
            'progress_percentage': (completed_stages / len(self.stages)) * 100,
            'total_attempts': total_attempts,
            'recent_performance': recent_performance,
            'learning_history_size': len(self.learning_history)
        }
    
    def reset_curriculum(self):
        """Reset the curriculum to start from beginning"""
        self.current_stage_index = 0
        for stage in self.stages:
            stage.completed = False
            stage.attempts = 0
            stage.best_performance = 0.0
        
        self.learning_history = []
        self.performance_trend = []
        print("🔄 Curriculum reset to stage 1")
    
    def get_recommended_scenario(self) -> Tuple[ScenarioType, Dict]:
        """Get recommended scenario based on current performance"""
        current_stage = self.get_current_stage()
        
        # If struggling, recommend easier scenario
        if (current_stage.attempts > 3 and 
            current_stage.best_performance < current_stage.performance_threshold * 0.8):
            # Try an easier scenario from available ones
            easier_scenarios = [s for s in current_stage.scenarios 
                              if s != ScenarioType.CONGESTION and s != ScenarioType.ACCIDENT]
            if easier_scenarios:
                scenario_type = random.choice(easier_scenarios)
            else:
                scenario_type = random.choice(current_stage.scenarios)
        else:
            # Normal random selection
            scenario_type = random.choice(current_stage.scenarios)
        
        parameters = self.scenario_generator.generate_scenario_parameters(
            scenario_type, current_stage.difficulty
        )
        
        return scenario_type, parameters

# Global curriculum manager instance
curriculum_manager = CurriculumManager()

def get_curriculum_manager() -> CurriculumManager:
    """Get the global curriculum manager instance"""
    return curriculum_manager

if __name__ == "__main__":
    # Test the curriculum system
    curriculum = CurriculumManager()
    
    print("🧪 Testing Curriculum Learning System:")
    print("=" * 50)
    
    # Test a few training sessions
    for i in range(3):
        print(f"\n--- Training Session {i+1} ---")
        
        # Start session
        session_params = curriculum.start_training_session()
        
        # Simulate performance metrics
        test_metrics = {
            'total_waiting_time': random.randint(50, 200),
            'vehicles_cleared': random.randint(15, 35),
            'average_speed': random.uniform(8, 15),
            'emergency_response_time': random.uniform(5, 25),
            'emergencies_handled': random.randint(0, 3)
        }
        
        # Update progress
        progress = curriculum.update_progress(test_metrics)
        
        print(f"   Performance Score: {progress['performance_score']:.3f}")
        print(f"   Stage Completed: {progress['stage_completed']}")
        print(f"   Curriculum Completed: {progress['curriculum_completed']}")
    
    # Show curriculum statistics
    stats = curriculum.get_curriculum_stats()
    print(f"\n📊 Curriculum Statistics:")
    print(f"   Progress: {stats['progress_percentage']:.1f}%")
    print(f"   Current Stage: {stats['current_stage']}")
    print(f"   Completed Stages: {stats['completed_stages']}/{stats['total_stages']}")
    print(f"   Recent Performance: {stats['recent_performance']:.3f}")