# Pet Nutrition AI Planner

A DSBA MLOps mini project for the **Pet** track. This system provides a **personalized diet planner** that adjusts food recommendations based on each pet’s profile, age, health conditions, and daily activity.

## 1. Problem Statement

Pet owners often feed their dogs or cats using generic package instructions or informal advice from social media. That creates three common problems:

- the portion size is not personalized to the pet’s **age, weight, neuter status, and activity level**
- owners do not know whether a specific food is **safe or unsafe**
- feeding decisions are rarely updated when the pet becomes more active, less active, older, or has health issues

This project addresses that gap with a practical AI application that combines pet profiling, rule-based calorie calculation, activity-aware meal adjustment, and a RAG-based nutrition advisor.

## 2. Solution Overview

The application has three main layers:

### 2.1 Pet profiling and data ingestion
- create user account and log in
- register pet information: name, species, breed, age, weight, neuter status
- store health conditions and food allergies
- log daily activity manually as steps and active minutes
- support future integration with wearable pet devices such as FitBark

### 2.2 Nutrition intelligence
- calculate **DER / daily calories** from pet profile
- generate **daily food portion and meal split**
- adjust food recommendation when activity changes
- answer nutrition questions with a **RAG-based chatbot** grounded in a curated knowledge base

### 2.3 User-facing application
- Gradio frontend for quick interaction
- FastAPI backend for API endpoints
- database-backed persistence for users, pets, sessions, and activity logs

## 3. Core Features

### Must-have
- pet registration and profile management
- daily calorie calculation
- nutrition chatbot with RAG
- activity-based plan adjustment

### Nice-to-have / future extensions
- automatic food subscription and reordering
- wearable device integration
- richer recommendation logic by breed, life stage, and medical constraints
- multi-agent workflow for more advanced planning and monitoring

## 4. Current System Architecture

```text
User
  -> Gradio Frontend
      -> FastAPI Backend
          -> Auth + Pet + Activity + Nutrition Routers
          -> Nutrition Calculator
          -> RAG Engine / LLM
          -> SQLAlchemy ORM
              -> SQLite (local dev) or Supabase Postgres
          -> Knowledge Base documents
```

## 5. Tech Stack

### Frontend
- Gradio

### Backend
- FastAPI
- SQLAlchemy
- Pydantic
- Uvicorn

### Database
- SQLite for local development
- Supabase Postgres for deployment

### AI / RAG
- LangChain
- ChromaDB
- Google Gemini API via `langchain-google-genai`
- PDF knowledge ingestion

### MLOps / Engineering
- Docker
- environment-based configuration with `.env`
- MLflow hooks for chat logging and experiment-style monitoring
- pytest for smoke/unit tests

## 6. Data Model

The backend currently uses only the tables that are actually needed by the running API:

- `users`
- `auth_sessions`
- `pets`
- `activity_logs`

Meal plans are generated **on demand** from the pet profile and latest activity, so a separate persisted `nutrition_plans` table is not required in the current implementation.

## 7. Main API Capabilities

### Authentication
- sign up
- log in
- log out
- validate current session

### Pet management
- create pet
- list user pets
- update pet profile
- delete pet profile

### Activity and diet adjustment
- log daily activity
- view recent activity history
- generate adjusted calorie and portion recommendation for a selected date

### Nutrition advisor
- ask food safety and nutrition questions
- return answers grounded in the project knowledge base
- personalize the answer with the selected pet profile

## 8. Why this project fits the DSBA Pet theme

This project clearly matches the course scope because it focuses on **pet healthcare and lifestyle**:

- healthcare: calorie planning, food safety, health conditions, allergies
- lifestyle: daily activity tracking and feeding adjustment
- AI application: RAG chatbot + rule-based recommendation engine
- MLOps angle: deployable API, reproducible setup, monitoring hooks, and containerization

## 9. How the project maps to the 7 presentation elements

### 1) Problem Statement
Owners lack trustworthy, personalized feeding guidance.

### 2) Solution
An AI-assisted pet nutrition planner with pet profiling, activity logs, calorie calculation, and RAG chatbot.

### 3) Target Users
Primary users are pet owners who want simple and personalized feeding advice. Secondary users could include pet clinics, pet shops, or nutrition-focused pet services.

### 4) Social Impact & Assessment
Possible measurable outcomes:
- number of registered pets
- number of nutrition questions answered
- average response latency
- number of activity-adjusted plans generated
- user feedback score on usefulness and trust

### 5) Team
This section should explain who handles backend, frontend, RAG/knowledge base, deployment, testing, and presentation.

### 6) Sustainability
Potential paths:
- premium nutrition insights
- subscription food planning
- B2B offering for pet clinics or pet product sellers

### 7) Demo & Technical Implementation
The project already supports a realistic live demo:
- user logs in
- creates a pet profile
- logs activity
- generates a meal adjustment
- asks a nutrition question in the chatbot

## 10. Suggested Evaluation Metrics

### System / engineering metrics
- API response time
- chatbot latency
- uptime / successful request rate
- retrieval hit quality from RAG

### Product metrics
- percentage of successful user flows
- average time to get a recommendation
- number of repeated users or repeated questions

### AI quality metrics
- groundedness / answer faithfulness
- relevance of retrieved chunks
- correctness of calorie calculation
- user-rated helpfulness

## 11. Local Setup

### Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --port 8000
```

### Frontend
Point the Gradio app to the backend base URL, then run the frontend app separately.

## 12. Future Work

- connect a real pet wearable API
- add breed-specific nutrition logic
- store chat history and evaluation results
- add admin dashboard for monitoring usage and latency
- add Alembic migrations for production-grade schema management
- expand the knowledge base with stronger veterinary nutrition sources

## 13. Summary

This project is not just a chatbot. It is a small end-to-end AI product with:

- a usable frontend
- a deployable backend
- persistent user and pet data
- personalized nutrition logic
- RAG-based consultation capability
- room for MLOps improvement and product growth

For your DSBA section, this is well aligned with the **Pet healthcare/lifestyle** requirement and has a clear path to present both technical depth and real user value.
