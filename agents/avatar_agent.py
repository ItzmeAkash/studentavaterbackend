import logging
import os
from livekit.agents import JobContext, WorkerOptions, cli, Agent, AgentSession
from livekit.plugins import deepgram, silero, tavus, groq
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("avatar")
logger.setLevel(logging.INFO)

# Log environment variables (without exposing secrets)
logger.info("Environment check:")
logger.info(f"LIVEKIT_URL: {'Set' if os.getenv('LIVEKIT_URL') else 'Missing'}")
logger.info(
    f"DEEPGRAM_API_KEY: {'Set' if os.getenv('DEEPGRAM_API_KEY') else 'Missing'}"
)
logger.info(f"GROQ_API_KEY: {'Set' if os.getenv('GROQ_API_KEY') else 'Missing'}")
logger.info(f"TAVUS_API_KEY: {'Set' if os.getenv('TAVUS_API_KEY') else 'Missing'}")


class AvatarAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""Hi! You are Miss Sarah, a friendly and open-minded physics teacher. Greet students warmly, keep the tone conversational and encouraging, and focus on explaining physics concepts clearly and simply.

Your primary role is to teach physics to students and answer their questions and doubts. Break down complex physics concepts—such as mechanics, thermodynamics, electromagnetism, optics, and quantum physics—into simple, understandable language. Use real-world examples and analogies to make concepts relatable. Be patient, empathetic, and always encourage students to ask questions.

CRITICAL INSTRUCTION: After explaining any concept, teaching any topic, or answering any question, you MUST end your response by asking if everything is clear or if the student has any doubts. Use phrases like "Does that make sense?", "Is everything clear?", "Do you have any questions about this?", or "Would you like me to explain any part in more detail?". This is essential—always check for understanding after each explanation.

Be polite, friendly, and open-minded. Create a welcoming learning environment where students feel comfortable asking questions. Encourage curiosity and celebrate their learning progress.

Important: Please respond using only plain text without any special formatting characters such as asterisks (*), underscores (_), hashtags (#), or any other markdown symbols. Your responses will be spoken aloud, so use natural, conversational language without any formatting markers.""",
        )


async def entrypoint(ctx: JobContext):
    logger.info("Starting agent and avatar session")

    try:
        # Connect to the context first
        await ctx.connect()
        logger.info("Connected to LiveKit room")

        # Initialize the agent
        agent = AvatarAgent()
        logger.info("Agent initialized")

        # Initialize AgentSession with required components
        session = AgentSession(
            stt=deepgram.STT(model="nova-3", language="multi"),
            llm=groq.LLM(model="llama-3.3-70b-versatile"),
            tts=deepgram.TTS(),
            vad=silero.VAD.load(),
            # Uncomment and configure if turn_detection is needed
            # turn_detection=MultilingualModel(),
        )
        logger.info("AgentSession initialized")

        # Initialize avatar session
        avatar = tavus.AvatarSession(replica_id=os.getenv("REPLICA_ID"), persona_id=os.getenv("PERSONA_ID"))
        logger.info("Avatar session initialized")

        # Start avatar session first
        logger.info("Starting avatar session")
        await avatar.start(session, room=ctx.room)

        # Start agent session
        logger.info("Starting agent session")
        await session.start(room=ctx.room, agent=agent)

        logger.info("All sessions started successfully")

    except Exception as e:
        logger.error(f"Error starting sessions: {e}")
        raise


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
