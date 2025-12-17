"""
BENCHMARKING SYSTEM FOR FITCOACH-LLM
=====================================
Evaluates workout plan generation quality using multiple metrics.
Compares different configurations to measure improvement.
"""

import json
import pandas as pd
from typing import Dict, List, Tuple
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import re
from datetime import datetime
import os


class WorkoutBenchmark:
    """Benchmark system for evaluating workout plan quality."""
    
    def __init__(self):
        self.model = ChatMistralAI(model="mistral-small-latest", temperature=0.3)
        self.output_parser = StrOutputParser()
        
    def create_test_cases(self) -> List[Dict]:
        """Create diverse test cases representing different user profiles."""
        return [
            {
                "name": "beginner_weight_loss",
                "gender": "Male ♂️",
                "age": "35",
                "height": "180",
                "weight": "95",
                "goals": "Weight loss",
                "target_zones": "Legs 🦵, Abs 🍫",
                "daily_time": "45",
                "days_per_week": "3",
                "equipment": "No equipment (bodyweight / home workouts only)",
                "user_notes": "Complete beginner, sedentary lifestyle, want to lose 10kg"
            },
            {
                "name": "intermediate_hypertrophy",
                "gender": "Female ♀️",
                "age": "28",
                "height": "165",
                "weight": "60",
                "goals": "Muscle gain (Hypertrophy)",
                "target_zones": "Legs 🦵, Back 💪, Shoulders 🤷‍♂️, Chest 🏋️, Arms 💪, Abs 🍫",
                "daily_time": "75",
                "days_per_week": "4",
                "equipment": "Dumbbells, Barbell, Bench, Gym machines",
                "user_notes": "6 months training experience, want to build muscle mass, especially upper body"
            },
            {
                "name": "senior_mobility",
                "gender": "Male ♂️",
                "age": "68",
                "height": "172",
                "weight": "78",
                "goals": "Flexibility",
                "target_zones": "Legs 🦵, Back 💪, Shoulders 🤷‍♂️",
                "daily_time": "30",
                "days_per_week": "5",
                "equipment": "Resistance bands, Pull-up bar",
                "user_notes": "Retired, mild arthritis in knees, want to maintain independence and mobility"
            },
            {
                "name": "athlete_endurance",
                "gender": "Female ♀️",
                "age": "24",
                "height": "170",
                "weight": "58",
                "goals": "Endurance improvement",
                "target_zones": "Legs 🦵, Abs 🍫",
                "daily_time": "60",
                "days_per_week": "6",
                "equipment": "Dumbbells, Kettlebell, TRX / suspension trainer",
                "user_notes": "Marathon runner, want to improve running economy and prevent injuries"
            },
            {
                "name": "busy_professional",
                "gender": "Male ♂️",
                "age": "42",
                "height": "175",
                "weight": "82",
                "goals": "General fitness",
                "target_zones": "Legs 🦵, Back 💪, Chest 🏋️, Abs 🍫",
                "daily_time": "30",
                "days_per_week": "3",
                "equipment": "Dumbbells, Resistance bands",
                "user_notes": "Desk job, limited time, mild lower back pain, want to stay healthy"
            }
        ]
    
    def evaluate_workout_plan(self, workout_plan: str, test_case: Dict) -> Dict[str, float]:
        """
        Evaluate a workout plan using multiple metrics.
        Returns scores between 0 and 1 for each metric.
        """
        metrics = {}
        
        # 1. Completeness Score: Check for essential components
        metrics['completeness'] = self._score_completeness(workout_plan)
        
        # 2. Structure Score: Check organization and formatting
        metrics['structure'] = self._score_structure(workout_plan)
        
        # 3. Personalization Score: Check alignment with user profile
        metrics['personalization'] = self._score_personalization(workout_plan, test_case)
        
        # 4. Safety Score: Check for warm-up, cool-down, progression
        metrics['safety'] = self._score_safety(workout_plan)
        
        # 5. Exercise Diversity: Check variety of exercises
        metrics['diversity'] = self._score_diversity(workout_plan)
        
        # 6. Specificity Score: Check target zones coverage
        metrics['specificity'] = self._score_specificity(workout_plan, test_case)
        
        # Overall score (weighted average)
        weights = {
            'completeness': 0.20,
            'structure': 0.15,
            'personalization': 0.25,
            'safety': 0.20,
            'diversity': 0.10,
            'specificity': 0.10
        }
        
        metrics['overall'] = sum(metrics[k] * weights[k] for k in weights.keys())
        
        return metrics
    
    def _score_completeness(self, plan: str) -> float:
        """Check if plan contains all essential components."""
        required_keywords = [
            'warm-up', 'cool-down', 'sets', 'reps', 'rest',
            'day', 'exercise', 'minutes'
        ]
        
        plan_lower = plan.lower()
        found = sum(1 for kw in required_keywords if kw in plan_lower)
        return found / len(required_keywords)
    
    def _score_structure(self, plan: str) -> float:
        """Check if plan is well-structured with clear days/sections."""
        score = 0.0
        
        # Check for day divisions
        day_patterns = r'(day\s+\d+|monday|tuesday|wednesday|thursday|friday|saturday|sunday)'
        days_found = len(re.findall(day_patterns, plan.lower()))
        if days_found >= 2:
            score += 0.4
        
        # Check for section headers (warm-up, main, cool-down)
        if 'warm' in plan.lower():
            score += 0.2
        if 'cool' in plan.lower():
            score += 0.2
        
        # Check for canonical exercise names
        canonical_pattern = r'canonical exercise name:'
        if re.search(canonical_pattern, plan.lower()):
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_personalization(self, plan: str, test_case: Dict) -> float:
        """Check if plan is personalized to user profile."""
        score = 0.0
        plan_lower = plan.lower()
        
        # Check goal alignment
        goal = test_case['goals'].lower()
        if 'weight loss' in goal and any(kw in plan_lower for kw in ['cardio', 'hiit', 'circuit']):
            score += 0.25
        elif 'hypertrophy' in goal and any(kw in plan_lower for kw in ['8-12', '3-4 sets', 'progressive']):
            score += 0.25
        elif 'flexibility' in goal and any(kw in plan_lower for kw in ['stretch', 'mobility', 'yoga']):
            score += 0.25
        elif 'endurance' in goal and any(kw in plan_lower for kw in ['cardio', 'stamina', 'aerobic']):
            score += 0.25
        else:
            score += 0.15
        
        # Check age considerations
        age = int(test_case['age'])
        if age > 60 and any(kw in plan_lower for kw in ['gentle', 'low-impact', 'mobility', 'balance']):
            score += 0.25
        elif age < 30 and any(kw in plan_lower for kw in ['intensity', 'explosive', 'advanced']):
            score += 0.25
        else:
            score += 0.15
        
        # Check equipment usage
        if 'no equipment' in test_case['equipment'].lower():
            if any(kw in plan_lower for kw in ['bodyweight', 'push-up', 'squat', 'plank']):
                score += 0.25
        else:
            equipment_items = test_case['equipment'].lower().split(',')
            equipment_mentioned = sum(1 for item in equipment_items if item.strip() in plan_lower)
            score += 0.25 * min(equipment_mentioned / max(len(equipment_items), 1), 1.0)
        
        # Check user notes consideration
        if test_case['user_notes']:
            notes_lower = test_case['user_notes'].lower()
            relevant_terms = []
            if 'beginner' in notes_lower:
                relevant_terms.append('beginner')
            if 'pain' in notes_lower or 'injury' in notes_lower:
                relevant_terms.extend(['careful', 'avoid', 'modify', 'gentle'])
            
            if any(term in plan_lower for term in relevant_terms):
                score += 0.25
            else:
                score += 0.1
        
        return min(score, 1.0)
    
    def _score_safety(self, plan: str) -> float:
        """Check for safety considerations."""
        score = 0.0
        plan_lower = plan.lower()
        
        # Warm-up presence
        if 'warm' in plan_lower:
            score += 0.3
        
        # Cool-down presence
        if 'cool' in plan_lower:
            score += 0.3
        
        # Rest periods mentioned
        if 'rest' in plan_lower or 'recovery' in plan_lower:
            score += 0.2
        
        # Progressive overload or gradual progression
        if any(kw in plan_lower for kw in ['progress', 'gradual', 'increase']):
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_diversity(self, plan: str) -> float:
        """Check exercise variety."""
        # Extract canonical exercise names
        pattern = r'canonical exercise name:\s*([^\n]+)'
        exercises = re.findall(pattern, plan.lower())
        
        if not exercises:
            return 0.3  # Base score if no canonical names found
        
        unique_exercises = len(set(exercises))
        
        # Score based on number of unique exercises
        if unique_exercises >= 10:
            return 1.0
        elif unique_exercises >= 7:
            return 0.8
        elif unique_exercises >= 5:
            return 0.6
        elif unique_exercises >= 3:
            return 0.4
        else:
            return 0.2
    
    def _score_specificity(self, plan: str, test_case: Dict) -> float:
        """Check if target zones are addressed."""
        target_zones_raw = test_case['target_zones']
        target_zones = [z.split()[0].lower() for z in target_zones_raw.split(',')]
        
        plan_lower = plan.lower()
        
        # Map zone names to exercise keywords
        zone_keywords = {
            'legs': ['squat', 'lunge', 'leg', 'quadriceps', 'hamstring', 'calf'],
            'back': ['row', 'pull', 'lat', 'back', 'deadlift'],
            'shoulders': ['shoulder', 'overhead', 'lateral', 'deltoid', 'press'],
            'chest': ['chest', 'bench', 'push-up', 'pec', 'press'],
            'arms': ['bicep', 'tricep', 'curl', 'arm', 'extension'],
            'abs': ['core', 'plank', 'crunch', 'ab', 'sit-up']
        }
        
        zones_covered = 0
        for zone in target_zones:
            keywords = zone_keywords.get(zone, [])
            if any(kw in plan_lower for kw in keywords):
                zones_covered += 1
        
        return zones_covered / max(len(target_zones), 1)
    
    def run_benchmark(self, prompt_template: PromptTemplate, config_name: str) -> pd.DataFrame:
        """
        Run benchmark on all test cases with a specific configuration.
        Returns DataFrame with results.
        """
        test_cases = self.create_test_cases()
        results = []
        
        chain = prompt_template | self.model | self.output_parser
        
        for test_case in test_cases:
            print(f"Evaluating: {test_case['name']}...")
            
            # Generate workout plan
            workout_plan = chain.invoke({
                "gender": test_case['gender'],
                "age": test_case['age'],
                "height": test_case['height'],
                "weight": test_case['weight'],
                "goals": test_case['goals'],
                "target_zones": test_case['target_zones'],
                "daily_time": test_case['daily_time'],
                "days_per_week": test_case['days_per_week'],
                "equipment": test_case['equipment'],
                "user_notes": test_case['user_notes']
            })
            
            # Evaluate the plan
            metrics = self.evaluate_workout_plan(workout_plan, test_case)
            
            # Store results
            result = {
                'config': config_name,
                'test_case': test_case['name'],
                **metrics
            }
            results.append(result)
        
        df = pd.DataFrame(results)
        return df
    
    def compare_configurations(self, configs: Dict[str, PromptTemplate]) -> pd.DataFrame:
        """
        Compare multiple prompt configurations.
        configs: Dict mapping config_name -> PromptTemplate
        """
        all_results = []
        
        for config_name, prompt_template in configs.items():
            print(f"\n{'='*60}")
            print(f"Running benchmark for: {config_name}")
            print(f"{'='*60}")
            
            df = self.run_benchmark(prompt_template, config_name)
            all_results.append(df)
        
        combined = pd.concat(all_results, ignore_index=True)
        return combined
    
    def generate_report(self, results_df: pd.DataFrame, output_path: str = "benchmark_results"):
        """Generate comprehensive benchmark report."""
        os.makedirs(output_path, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        csv_path = f"{output_path}/results_{timestamp}.csv"
        results_df.to_csv(csv_path, index=False)
        
        # Generate summary statistics
        summary = results_df.groupby('config').agg({
            'overall': ['mean', 'std'],
            'completeness': 'mean',
            'structure': 'mean',
            'personalization': 'mean',
            'safety': 'mean',
            'diversity': 'mean',
            'specificity': 'mean'
        }).round(3)
        
        summary_path = f"{output_path}/summary_{timestamp}.csv"
        summary.to_csv(summary_path)
        
        # Generate text report
        report_path = f"{output_path}/report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write("FitCoach-LLM Benchmark Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Summary Statistics by Configuration:\n")
            f.write("-" * 60 + "\n")
            f.write(summary.to_string())
            f.write("\n\n")
            
            f.write("Detailed Results by Test Case:\n")
            f.write("-" * 60 + "\n")
            for config in results_df['config'].unique():
                f.write(f"\n{config}:\n")
                config_df = results_df[results_df['config'] == config]
                f.write(config_df[['test_case', 'overall', 'completeness', 
                                   'personalization', 'safety']].to_string(index=False))
                f.write("\n")
        
        print(f"\nBenchmark results saved to: {output_path}/")
        print(f"  - Raw results: {csv_path}")
        print(f"  - Summary: {summary_path}")
        print(f"  - Report: {report_path}")
        
        return summary


# Example usage
if __name__ == "__main__":
    from prompt import workout_prompt
    
    # Create benchmark instance
    benchmark = WorkoutBenchmark()
    
    # Define configurations to compare
    # You can add more configurations with different prompts
    configs = {
        "baseline": workout_prompt,
        # Add more configurations here as you improve the system
    }
    
    # Run comparison
    results = benchmark.compare_configurations(configs)
    
    # Generate report
    summary = benchmark.generate_report(results)
    
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(summary)