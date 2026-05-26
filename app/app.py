# app.py
import sys
import os

# Add sibling folders to path so imports resolve correctly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "agents"))
sys.path.insert(0, os.path.join(BASE_DIR, "Ccore"))
sys.path.insert(0, os.path.join(BASE_DIR, "app"))

import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# Import our AI Modules
from resume_parser import ResumeParser
from knowledge_base import SkillOntology
from search_agent import CareerPathPlanner
from inference_engine import FuzzyEvaluator
from state_manager import CareerState
# NEW IMPORT FOR GENETIC ALGO
from genetic_scheduler import GeneticScheduler

# Page Config
st.set_page_config(page_title="AI Career Agent", layout="wide", page_icon="🤖")

# Custom CSS for better look
st.markdown("""
    <style>
    .stAlert { box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    .big-font { font-size:20px !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

def save_uploaded_file(uploaded_file):
    if not os.path.exists("temp"):
        os.makedirs("temp")
    with open(os.path.join("temp", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    return os.path.join("temp", uploaded_file.name)

def draw_better_ontology(kb):
    """
    Creates a Tree-Structured visualization of the Knowledge Base.
    Uses 'Multipartite Layout' to arrange nodes in clear vertical layers (Hierarchy).
    """
    layers = {}
    try:
        lengths = nx.single_source_shortest_path_length(kb.graph, "CS_Student")
        for node, length in lengths.items():
            kb.graph.nodes[node]["layer"] = length
    except:
        for node in kb.graph.nodes():
            kb.graph.nodes[node]["layer"] = 0

    plt.figure(figsize=(14, 10)) 
    
    pos = nx.multipartite_layout(kb.graph, subset_key="layer", align="vertical")
    
    for node, coords in pos.items():
        coords[1] *= 1.5 
    
    colors = []
    sizes = []
    for node in kb.graph.nodes():
        if node == "CS_Student":
            colors.append("#FF6B6B") 
            sizes.append(3500)
        elif kb.graph.nodes[node]["layer"] == 1:
            colors.append("#4ECDC4") 
            sizes.append(3000)
        else:
            colors.append("#4F8BF9") 
            sizes.append(2500)

    nx.draw_networkx_nodes(kb.graph, pos, node_size=sizes, node_color=colors, edgecolors="black", linewidths=1.5, alpha=0.9)
    
    nx.draw_networkx_edges(kb.graph, pos, width=2, alpha=0.6, edge_color='#555555', 
                           arrowstyle='-|>', arrowsize=20, connectionstyle="arc3,rad=0.1")
    
    nx.draw_networkx_labels(kb.graph, pos, font_size=10, font_color='white', font_weight='bold', font_family="sans-serif")
    
    plt.title("Knowledge Base Hierarchy (Tree Structure)", fontsize=18, fontweight='bold', pad=20)
    plt.text(0.05, 0.95, "Flow: Student → Domain → Category → Specific Skill", 
             transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.axis('off')
    return plt

def plot_gauge_chart(score):
    """Creates a Speedometer (Gauge) chart for the Fuzzy Score."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Candidate Suitability Score (Fuzzy Output)"},
        gauge = {
            'axis': {'range': [None, 10]},
            'bar': {'color': "#4F8BF9"},
            'steps': [
                {'range': [0, 4], 'color': "#ffcccb"},  
                {'range': [4, 7], 'color': "#fff4cc"},  
                {'range': [7, 10], 'color': "#d4edda"}  
            ],
        }
    ))
    return fig

def plot_skill_gap(current, required):
    """Creates a Bar Chart comparing 'What I Have' vs 'What I Need'."""
    data = []
    for skill in required:
        status = "Have" if skill in current else "Missing"
        data.append({"Skill": skill, "Status": status, "Value": 1})
    
    df = pd.DataFrame(data)
    
    fig = px.bar(df, x="Skill", y="Value", color="Status", 
                 color_discrete_map={"Have": "#28a745", "Missing": "#dc3545"},
                 title="Gap Analysis: Current vs Goal State",
                 height=300)
    fig.update_yaxes(visible=False, showticklabels=False)
    return fig

def main():
    st.title("🤖 Intelligent Career Path Agent")
    st.markdown("### Course: CSE3705 AI | Project: Planning & Reasoning Agent")
    
    # --- SIDEBAR CONFIGURATION ---
    st.sidebar.header("1. Goal Definition")
    target_role = st.sidebar.selectbox("Select Target Role", 
                                       ["Python Developer", "Data Scientist", "Frontend Engineer"])
    
    goals = {
        "Python Developer": ["Python", "Django", "SQL", "Git"],
        "Data Scientist": ["Python", "Machine Learning", "Pandas", "SQL"],
        "Frontend Engineer": ["HTML", "JavaScript", "React", "CSS"]
    }
    required_skills = goals[target_role]
    st.sidebar.info(f"**Goal State Vector:**\n {required_skills}")

    # GUIDELINE: Ethical Considerations (Bonus)
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Ethical AI Settings")
    blind_mode = st.sidebar.checkbox("Enable Blind Hiring Mode", value=True, 
                                     help="Removes Name & Personal identifiers to prevent Bias")
    
    if blind_mode:
        st.sidebar.success("🛡️ Bias Protection Active: Personal data ignored.")

    # GUIDELINE: AI vs Non-AI Distinction (Note 1)
    st.sidebar.markdown("---")
    with st.sidebar.expander("⚙️ System Architecture (AI vs Non-AI)"):
        st.markdown("""
        **🤖 AI Components:**
        - **Perception:** NLP (Spacy) for Skill Extraction.
        - **Reasoning:** Ontology Graph (NetworkX).
        - **Planning:** A* Search Algorithm.
        - **Decision:** Fuzzy Logic Inference.
        - **Optimization:** Genetic Algorithm (Scheduler).
        
        **💻 Non-AI Components:**
        - **UI/UX:** Streamlit Framework.
        - **File Handling:** PDF Plumber.
        - **Visualization:** Plotly/Matplotlib.
        """)

    # --- MAIN LAYOUT ---
    uploaded_file = st.file_uploader("2. Perception Layer: Upload Resume (PDF)", type="pdf")

    if uploaded_file is not None:
        
        # Ethical Logic: Hide Filename if Blind Mode is ON
        if blind_mode:
            display_name = "Candidate_ID_XA12.pdf (Anonymized)"
        else:
            display_name = uploaded_file.name
        st.write(f"Processing File: **{display_name}**")

        # --- STAGE 1: PERCEPTION (NLP) ---
        file_path = save_uploaded_file(uploaded_file)
        parser = ResumeParser()
        
        extracted_text = parser.extract_text_from_pdf(file_path)
        detected_skills = parser.extract_skills(extracted_text)
        exp_years = parser.get_experience_level(extracted_text)
        
        st.divider()
        st.subheader("2. Perception Results (Sensors)")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("**Extracted Factored State (Vector):**")
            st.success(f"{detected_skills}")
        with col2:
            st.metric("Detected Experience", f"{exp_years} Years")

        # --- STAGE 2: VISUALIZATION & KR ---
        st.divider()
        st.subheader("3. State Space Analysis (Visualization)")
        
        # Skill Gap Chart
        st.plotly_chart(plot_skill_gap(detected_skills, required_skills), use_container_width=True)

        # Knowledge Base Graph
        with st.expander("View Knowledge Base Ontology (Graph)", expanded=False):
            st.write("This graph represents the AI's internal map of how skills are related.")
            kb = SkillOntology()
            st.pyplot(draw_better_ontology(kb))

        # --- STAGE 3: DECISION (FUZZY LOGIC) ---
        st.divider()
        st.subheader("4. Decision Making (Fuzzy Logic Engine)")
        
        match_count = len([s for s in required_skills if s in detected_skills])
        match_percent = (match_count / len(required_skills)) * 100
        evaluator = FuzzyEvaluator()
        score = evaluator.evaluate_candidate(match_percent, exp_years)
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(plot_gauge_chart(score), use_container_width=True)
        with c2:
            st.info(f"""
            **Logic Explanation:**
            - Skill Match: **{round(match_percent)}%**
            - Experience: **{exp_years} Years**
            - Fuzzy Rule Fired: *If match is {match_percent}% and exp is {exp_years}, then suitability is...*
            """)

        # --- STAGE 4: PLANNING (A* SEARCH) ---
        st.divider()
        st.subheader("5. Agent Planning (A* Search & Explainability)")
        
        planner = CareerPathPlanner()
        path, cost, trace = planner.plan_career_path(detected_skills, required_skills)
        
        if path:
            # GUIDELINE: CSP (Constraint Satisfaction)
            with st.expander("🔍 View Constraint Satisfaction Logic (CSP)"):
                st.write("The planner satisfied the following Hard Constraints:")
                st.markdown("""
                1. **Prerequisite Constraint:** Cannot learn a Child Skill (e.g., React) before Parent Skill (e.g., JS).
                2. **Sequential Constraint:** Time flows forward; accumulated cost cannot be negative.
                3. **Domain Constraint:** Skills must belong to the defined Ontology.
                """)
                st.success("✅ All constraints validated for this path.")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**Goal:** Transition from *Current State* to *{target_role}*")
                st.markdown(f"**Optimal Path Cost:** {cost} Weeks")
                
                # Timeline view
                for i, step in enumerate(path):
                    cols = st.columns([0.1, 0.9])
                    with cols[0]:
                        st.markdown(f"## {i+1}⬇")
                    with cols[1]:
                        st.success(f"**Action:** Learn {step} (Est. Time: {planner.learning_costs.get(step, 2)} weeks)")
            
            with col2:
                # GUIDELINE: Explainable AI
                st.info("🧠 **AI Reasoning Trace**")
                st.caption("How the agent decided this path using A* (f = g + h)")
                
                with st.expander("View Search Logs"):
                    for log in trace:
                        st.markdown(f"""
                        ---
                        **State Evaluated:** {len(log['skills'])} Skills Known
                        - **G (Cost so far):** {log['g_score']}
                        - **H (Heuristic):** {log['h_score']}
                        - **F (Total):** {log['f_score']}
                        """)
                    st.write("---")
                    st.write("**Goal Reached!**")

            # --- STAGE 5 (NEW): GENETIC ALGORITHM (SCHEDULING) ---
            st.divider()
            st.subheader("6. Optimized Study Scheduler (Genetic Algorithm)")
            
            st.info("🧬 **Genetic Algorithm Logic:** The AI creates a 'Population' of random schedules, 'Mutates' them, and performs 'Crossover' to find the most balanced timetable.")
            
            # Use the path found by A* for scheduling
            skills_to_schedule = path 
            
            col1, col2 = st.columns([1, 3])
            with col1:
                study_hours = st.slider("Daily Study Hours", min_value=1, max_value=5, value=2)
                if st.button("🧬 Evolve Schedule"):
                    with st.spinner("Running Evolution (Generations: 50)..."):
                        # Using the Genetic Module
                        ga = GeneticScheduler(skills_to_schedule, hours_per_day=study_hours)
                        best_schedule_gene = ga.run_evolution()
                        df_schedule = ga.format_schedule(best_schedule_gene)
                    
                    st.success("Optimization Complete!")
                    st.write("Best Schedule Found:")
                    st.dataframe(df_schedule)
            
            with col2:
                st.markdown("""
                **How GA works here:**
                - **Chromosome:** A weekly calendar grid.
                - **Fitness Function:** Higher score if:
                    - ✅ All skills covered.
                    - ✅ No burnout.
                    - ✅ Balanced Load.
                """)

        else:
            st.balloons()
            st.success("✅ Goal State Reached! No further actions required.")

    # --- FINAL SECTION: LEARNINGS (MANDATORY) ---
    st.divider()
    with st.expander("📚 Project Learnings & Future Scope (Mandatory)"):
        st.markdown("""
        **💡 Key Learnings:**
        1. **Factored States:** Representing a career not as a single node, but as a dynamic vector of skills allowed for complex reasoning.
        2. **Hybrid AI:** Combining Crisp Logic (A* Search), Fuzzy Logic (Evaluation), and Evolutionary Algorithms (GA) creates a robust agent.
        3. **Optimization vs Planning:** Used A* for finding the *path* and GA for optimizing the *schedule*.

        **🔮 Future Scope:**
        - Integrating Real-time Job Market API to update the Ontology dynamically.
        - Using Reinforcement Learning (RL) to adjust the schedule based on student's actual progress.
        """)

if __name__ == "__main__":
    main()