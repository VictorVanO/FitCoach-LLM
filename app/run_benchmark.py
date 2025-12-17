"""
BENCHMARK RUNNER
================
Run this script to evaluate different configurations of your FitCoach-LLM system.
"""

from langchain_core.prompts import PromptTemplate
from benchmark_system import WorkoutBenchmark
from prompt import workout_prompt
from dotenv import load_dotenv

load_dotenv(override=True)


# Define different prompt configurations to compare
def get_baseline_prompt():
    """Original prompt without advanced techniques."""
    return workout_prompt


def get_enhanced_prompt():
    """Enhanced prompt with more detailed instructions."""
    enhanced_text = """
You are an expert workout coach with 10 years of experience and certifications in:
- Exercise Physiology
- Sports Science
- Injury Prevention
- Program Design

Please create a COMPREHENSIVE workout program with the following user information:

USER PROFILE:
-------------
Gender: {gender}
Age: {age} years
Height: {height} cm
Weight: {weight} kg
Primary Goals: {goals}
Target Zones: {target_zones}
Available Time: {daily_time} minutes per session
Training Frequency: {days_per_week} days per week
Equipment: {equipment}
Additional Notes: {user_notes}

PROGRAM DESIGN REQUIREMENTS:
----------------------------
1. WARM-UP (5-10 minutes):
   - Dynamic stretches
   - Activation exercises
   - Heart rate preparation

2. MAIN WORKOUT:
   - 3-5 compound or isolation exercises per zone
   - Sets: 2-4 sets per exercise
   - Reps: Appropriate for goal (strength: 4-6, hypertrophy: 8-12, endurance: 15-20)
   - Rest: 30-90 seconds (goal-dependent)
   - Tempo: Controlled eccentric, explosive concentric

3. COOL-DOWN (5-10 minutes):
   - Static stretching
   - Foam rolling suggestions
   - Breathing exercises

4. SAFETY & PROGRESSION:
   - Age-appropriate modifications
   - Injury prevention cues
   - Progressive overload strategy
   - Deload week recommendations

5. EXERCISE SELECTION:
   - Balance push/pull movements
   - Include unilateral exercises
   - Core stability throughout
   - Functional movement patterns

CRITICAL: After EACH exercise, output on a new line:
Canonical Exercise Name: <simple standard name>

Examples:
- "Barbell Back Squat" → Canonical Exercise Name: Squat
- "Dumbbell Chest Press" → Canonical Exercise Name: Bench Press
- "Bodyweight Push-Up" → Canonical Exercise Name: Push-up

FORMAT YOUR RESPONSE AS:
========================

**Weekly Overview:**
- Brief 2-3 sentence summary of the program structure

**Day 1: [Focus Area]**

*Warm-up (5-10 min):*
1. [Exercise] - [Duration/Reps]
2. [Exercise] - [Duration/Reps]

*Main Workout:*
1. [Exercise Name] - [Sets] x [Reps] @ [Rest]
   Canonical Exercise Name: [Name]
   - Form cues: [Key points]

2. [Exercise Name] - [Sets] x [Reps] @ [Rest]
   Canonical Exercise Name: [Name]
   - Form cues: [Key points]

*Cool-down (5-10 min):*
1. [Stretch] - [Duration]
2. [Stretch] - [Duration]

[Repeat for each training day]

**Progression Strategy:**
- Week 1-2: [Guidelines]
- Week 3-4: [Guidelines]

**Safety Notes:**
- [Age/injury considerations]
- [Warning signs to watch for]
"""
    return PromptTemplate.from_template(enhanced_text)


def get_concise_prompt():
    """Shorter, more efficient prompt for comparison."""
    concise_text = """
Expert coach: Create {days_per_week}-day workout for:
- {age}yo {gender}, {height}cm, {weight}kg
- Goal: {goals}
- Zones: {target_zones}
- Time: {daily_time}min/day
- Gear: {equipment}
- Notes: {user_notes}

Each day include:
1. Warm-up (5min)
2. 4-5 main exercises (sets×reps, rest time)
3. Cool-down (5min)

After each exercise add:
Canonical Exercise Name: [standard name]

Focus on safety, progression, personalization.
"""
    return PromptTemplate.from_template(concise_text)


def main():
    """Run benchmark comparison."""
    print("="*70)
    print("FitCoach-LLM BENCHMARK EVALUATION")
    print("="*70)
    print("\nThis will evaluate three prompt configurations:")
    print("1. Baseline: Original prompt")
    print("2. Enhanced: Detailed instructions with structure")
    print("3. Concise: Minimal tokens, efficiency-focused")
    print("\nEach configuration will be tested on 5 diverse user profiles.")
    print("="*70)
    
    input("\nPress Enter to start benchmark (this will take several minutes)...")
    
    # Initialize benchmark
    benchmark = WorkoutBenchmark()
    
    # Define configurations
    configs = {
        "baseline": get_baseline_prompt(),
        "enhanced": get_enhanced_prompt(),
        "concise": get_concise_prompt(),
    }
    
    # Run comparison
    print("\nRunning benchmark...")
    results = benchmark.compare_configurations(configs)
    
    # Generate comprehensive report
    print("\nGenerating report...")
    summary = benchmark.generate_report(results)
    
    # Display results
    print("\n" + "="*70)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*70)
    print("\nAverage Scores by Configuration:")
    print("-"*70)
    print(summary)
    
    # Find best configuration
    overall_means = results.groupby('config')['overall'].mean()
    best_config = overall_means.idxmax()
    best_score = overall_means.max()
    
    print("\n" + "="*70)
    print(f"BEST CONFIGURATION: {best_config}")
    print(f"Overall Score: {best_score:.3f}")
    print("="*70)
    
    # Detailed breakdown
    print("\nDetailed Metric Breakdown:")
    print("-"*70)
    metrics_of_interest = ['completeness', 'personalization', 'safety', 'diversity']
    for metric in metrics_of_interest:
        print(f"\n{metric.upper()}:")
        metric_scores = results.groupby('config')[metric].mean().sort_values(ascending=False)
        for config, score in metric_scores.items():
            print(f"  {config:15s}: {score:.3f}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nFull results saved to: ./benchmark_results/")
    print("\nUse these metrics in your report to demonstrate improvement!")


if __name__ == "__main__":
    main()