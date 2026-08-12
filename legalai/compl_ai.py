"""COMPL-AI Interactive Compliance Flows and Diagnostics."""

COMPL_AI_FLOWS = {
    "gpai_provider": (
        "### COMPL-AI Interactive Flow: Am I a GPAI Provider?\n\n"
        "To determine if you qualify as a General Purpose AI (GPAI) provider under the EU AI Act (COMPL-AI framework), "
        "evaluate this diagnostic tree. Please reply with details about your model:\n\n"
        "1. **Generality Test**: Does your model display significant generality and is it capable of competently "
        "performing a wide range of distinct tasks (e.g., text synthesis, translation, coding, reasoning)?\n"
        "   - *If Yes, continue. If No, you are likely not a GPAI provider (your system is a narrow AI system).* \n\n"
        "2. **Release/Market Test**: Have you placed the model on the EU market, or put it into service under your own "
        "brand or name (either as a standalone API, open-source download, or integrated system)?\n"
        "   - *If Yes, you are a GPAI provider. If you only use an external API (like OpenAI), you are a **Deployer**.* \n\n"
        "3. **Systemic Risk Threshold (Compute)**: Did the cumulative compute used to train your model exceed **10^25 FLOPs**?\n"
        "   - *If Yes, your model is classified as a **GPAI Model with Systemic Risk** (Tier 3).* \n"
        "   - *If No, it is a standard GPAI model (Tier 1 or Tier 2 depending on licensing).* \n\n"
        "**Action Required**: Reply with your model's **training compute estimate (FLOPs)** and **licensing model (Open/Closed)** to get an automated status determination."
    ),
    "obligations": (
        "### COMPL-AI Interactive Flow: Which Obligations Apply to Me?\n\n"
        "Under the COMPL-AI compliance framework, obligations are divided into three tiers depending on model classification:\n\n"
        "| Tier | Classification | Primary Obligations | Evidence Required |\n"
        "| :--- | :--- | :--- | :--- |\n"
        "| **Tier 1** | **Standard Open-Weight GPAI** | 1. Implement a policy to respect EU copyright law (opt-outs).<br>2. Publish a detailed summary of the content/datasets used for training. | - Copyright policy document<br>- Public training summary |\n"
        "| **Tier 2** | **Standard Commercial/Closed GPAI** | 1. Standard Tier 1 obligations.<br>2. Draw up and maintain detailed Technical Documentation.<br>3. Provide instructions & integration info to downstream deployers. | - Technical dossier (Annex VIII)<br>- Downstream API/use guides |\n"
        "| **Tier 3** | **GPAI with Systemic Risk** (Compute > 10^25 FLOPs) | 1. All Tier 1 & 2 obligations.<br>2. Perform model evaluation (adversarial red-teaming).<br>3. Track, document, and report serious incidents.<br>4. Implement cybersecurity protections. | - Red-teaming reports<br>- Incident logs & plans<br>- Cybersecurity audit certificates |\n\n"
        "**Action Required**: Tell me your model's licensing (Open/Closed) and compute class to identify your applicable requirements."
    ),
    "evidence": (
        "### COMPL-AI Interactive Flow: What Evidence Do I Need?\n\n"
        "To satisfy the compliance requirements under COMPL-AI, compile the following evidence packages:\n\n"
        "1. **Copyright Compliance Dossier**:\n"
        "   - Documented procedures for tracking machine-readable opt-outs (e.g., robots.txt tags, web-crawler flags).\n"
        "   - Documentation of dataset filtering algorithms used to exclude copyrighted materials.\n\n"
        "2. **Technical Documentation (Annex VIII)**:\n"
        "   - General description of the model (architecture, parameters, hardware setup, compute time).\n"
        "   - Detailed training protocols, dataset sourcing methods, and data governance policies.\n"
        "   - Detailed evaluation protocols, benchmark metrics (e.g., MMLU, LegalBench), and safety evaluations.\n\n"
        "3. **Downstream Deployment Package**:\n"
        "   - Technical sheets showing model capabilities, limits, inputs, and outputs.\n"
        "   - Instructions for downstream developers on how to prevent risks and maintain safety.\n\n"
        "4. **Systemic Risk Mitigation Package**:\n"
        "   - Adversarial red-teaming report demonstrating testing against safety vulnerabilities (jailbreaks, bias, cybersecurity threats).\n"
        "   - Incident response plan showing how serious model issues will be tracked and escalated.\n\n"
        "**Action Required**: Choose one of the compliance categories above, and I will help you draft the template for that evidence package."
    )
}

def get_compl_ai_response(query: str) -> str | None:
    """Return a matched COMPL-AI compliance response if query matches key terms."""
    q = query.lower()
    if "gpai provider" in q or "am i a gpai" in q:
        return COMPL_AI_FLOWS["gpai_provider"]
    elif "obligation" in q or "obligations apply" in q:
        return COMPL_AI_FLOWS["obligations"]
    elif "evidence" in q or "evidence package" in q or "what evidence" in q:
        return COMPL_AI_FLOWS["evidence"]
    elif "compl-ai" in q or "compl ai" in q:
        return (
            "### COMPL-AI Compliance Hub\n\n"
            "Welcome to the COMPL-AI interactive framework assistant. I can guide you through the following workflows:\n"
            "- **Am I a GPAI provider?**: Determine your regulatory status.\n"
            "- **Which obligations apply?**: Check applicable obligations.\n"
            "- **What evidence do I need?**: Discover evidence requirements for compliance audit preparation.\n\n"
            "Please select one of the topics above or ask a specific question!"
        )
    return None
