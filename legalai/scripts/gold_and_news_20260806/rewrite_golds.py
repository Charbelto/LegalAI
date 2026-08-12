"""Replace the 30 draft reference answers with citation-verified ones.

Every article number asserted below was checked against the text actually in
chroma_storage/collection_1 (644 chunks of CELEX:32024R1689), not from memory.
The drafts they replace were written by llama3.1:8b against the *stale news
corpus* (bug 8) and cite articles that say something else entirely: q03 claimed
Article 5 governs high-risk design (it lists prohibited practices), q04 claimed
the Act never mentions CE marking (Article 48 is titled "CE marking"), q29 quoted
GDPR's EUR 20M/4% instead of Article 99's EUR 35M/7%.

q21-q30 ask for news. The corpus is statute-only and the benchmark sends
fetch_news=False, so no news source exists. Their golds are rewritten to reward
what a correct system can actually do -- state the statutory position and say
plainly that live developments cannot be confirmed -- and stay flagged
needs_review because that reframing is a methodology decision, not a fix.
"""

import json
import shutil
from pathlib import Path

ROOT = Path(r"C:\Users\Charbel\Desktop\Legal AI\legalai")
DATASET = ROOT / "eval_dataset.json"
BACKUP = ROOT / "eval_dataset.draft_golds_backup_20260806.json"

REVIEWER = "claude-opus-5: every citation verified against corpus text (CELEX:32024R1689)"

# --- q01-q20: answerable from the Act corpus, citations verified -------------
VERIFIED = {

"q01": """**Issue:** What is a high-risk AI system under the EU AI Act?

**Rule:** Classification is governed by Article 6, not by the definitions in Article 3. An AI system is high-risk under either of two routes. Under Article 6(1), it is high-risk where it is intended to be used as a safety component of a product, or is itself a product, covered by the Union harmonisation legislation listed in Annex I, *and* that product is required to undergo a third-party conformity assessment under that legislation. Under Article 6(2), the AI systems listed in Annex III are high-risk. Annex III covers biometrics; critical infrastructure; education and vocational training; employment and worker management; access to essential private and public services and benefits; law enforcement; migration, asylum and border control; and the administration of justice and democratic processes.

**Application:** Article 6(3) provides a derogation: an Annex III system is not high-risk where it does not pose a significant risk of harm to health, safety or fundamental rights, including by not materially influencing the outcome of decision-making. This applies where the system performs a narrow procedural task, improves the result of a previously completed human activity, detects decision-making patterns without replacing or influencing human assessment, or performs a preparatory task. A system that performs profiling of natural persons is always high-risk. A provider relying on the derogation must document its assessment (Article 6(4)). Systems so classified are subject to the requirements in Chapter III Section 2 (Articles 8-15) and to provider and deployer obligations in Articles 16 and 26.

**Conclusion:** A high-risk AI system is one caught by Article 6(1) as a regulated product or safety component, or listed in Annex III under Article 6(2), unless the Article 6(3) derogation applies. Classification triggers Articles 8-15 and the obligations in Articles 16 and 26.""",

"q02": """**Issue:** What are the penalties for violating prohibited AI practices under Article 71 of the EU AI Act?

**Rule:** The premise of the question should be corrected. Article 71 establishes the EU database for high-risk AI systems listed in Annex III; it does not set penalties. Penalties are governed by Article 99. Under Article 99(3), non-compliance with the prohibition of the AI practices referred to in Article 5 is subject to administrative fines of up to EUR 35 000 000 or, if the offender is an undertaking, up to 7 % of its total worldwide annual turnover for the preceding financial year, whichever is higher.

**Application:** This is the Act's most severe tier. Other infringements, including breaches of provider or deployer obligations, attract fines under Article 99(4) of up to EUR 15 000 000 or 3 % of total worldwide annual turnover, whichever is higher. Supplying incorrect, incomplete or misleading information to notified bodies or national competent authorities attracts fines under Article 99(5) of up to EUR 7 500 000 or 1 %, whichever is higher. For SMEs, including start-ups, Article 99(6) provides that each fine is capped at the lower of the percentage or the fixed amount. Article 99(1) requires Member States to lay down rules on penalties that are effective, proportionate and dissuasive, and Article 99(7) lists the factors relevant to setting the amount, including the nature and gravity of the infringement and whether it was intentional or negligent. Fines on providers of general-purpose AI models are set separately by Article 101.

**Conclusion:** Penalties for prohibited practices come from Article 99(3), not Article 71: up to EUR 35 000 000 or 7 % of worldwide annual turnover, whichever is higher.""",

"q03": """**Issue:** What does Article 5 of the EU AI Act prohibit?

**Rule:** Article 5(1) prohibits placing on the market, putting into service, or using the following AI practices: (a) systems deploying subliminal techniques beyond a person's consciousness, or purposefully manipulative or deceptive techniques, with the object or effect of materially distorting behaviour by appreciably impairing the ability to make an informed decision, causing significant harm; (b) systems exploiting vulnerabilities due to age, disability, or a specific social or economic situation, with the same effect; (c) social scoring, meaning evaluation or classification of persons over time based on social behaviour or personal characteristics, leading to detrimental treatment in unrelated contexts or treatment that is unjustified or disproportionate; (d) assessing or predicting the risk of a person committing a criminal offence based solely on profiling or personality traits, except to support a human assessment already grounded in objective, verifiable facts directly linked to criminal activity; (e) untargeted scraping of facial images from the internet or CCTV to build facial recognition databases; (f) inferring emotions in the workplace or in education institutions, except for medical or safety reasons; (g) biometric categorisation to deduce race, political opinions, trade union membership, religious or philosophical beliefs, sex life or sexual orientation; and (h) real-time remote biometric identification in publicly accessible spaces for law enforcement, subject to narrow exceptions.

**Application:** The Article 5(1)(h) exceptions are limited to targeted searches for victims of abduction, trafficking or sexual exploitation and missing persons; prevention of a specific, substantial and imminent threat to life or physical safety, or a genuine and present or foreseeable threat of terrorist attack; and locating or identifying a suspect in the offences listed in Annex II punishable by a custodial sentence of at least four years. Such use requires a fundamental rights impact assessment, registration, and prior authorisation by a judicial or independent administrative authority, save in duly justified urgent cases.

**Conclusion:** Article 5 sets out the Act's unacceptable-risk tier: eight categories of prohibited practice, subject to strictly limited law enforcement exceptions with procedural safeguards.""",

"q04": """**Issue:** What is the CE marking requirement for high-risk AI systems?

**Rule:** The Act does address CE marking expressly: Article 48 is titled "CE marking". Article 48(1) provides that the CE marking is subject to the general principles in Article 30 of Regulation (EC) No 765/2008. Article 48(3) requires it to be affixed visibly, legibly and indelibly to the high-risk AI system, or, where that is not possible or not warranted by the nature of the system, to its packaging or accompanying documentation. Article 48(2) provides that for high-risk AI systems provided digitally, a digital CE marking is used only where it can be easily accessed via the interface from which the system is accessed or via an easily accessible machine-readable code or other electronic means. Article 48(4) requires the marking to be followed by the identification number of the notified body, where one was involved in the conformity assessment under Article 43, and that number must also appear in promotional material.

**Application:** Affixing the CE marking is a provider obligation under Article 16(i). It comes at the end of a sequence: the system must meet the Chapter III Section 2 requirements (Articles 8-15), undergo the applicable conformity assessment under Article 43, and be covered by an EU declaration of conformity drawn up under Article 47. Where the system is also covered by other Union law providing for CE marking, Article 48(5) provides that the marking indicates conformity with that other law as well.

**Conclusion:** CE marking is expressly required by Article 48, affixed by the provider under Article 16(i) after conformity assessment under Article 43 and the declaration of conformity under Article 47.""",

"q05": """**Issue:** What is the definition of a general-purpose AI (GPAI) model?

**Rule:** Article 3(63) defines a general-purpose AI model as an AI model, including where it is trained with a large amount of data using self-supervision at scale, that displays significant generality and is capable of competently performing a wide range of distinct tasks regardless of the way the model is placed on the market, and that can be integrated into a variety of downstream systems or applications. The definition expressly excludes AI models used for research, development or prototyping activities before they are placed on the market.

**Application:** The definition should be distinguished from two neighbouring concepts. A general-purpose AI *system* is defined separately in Article 3(66) as a system based on such a model that has the capability to serve a variety of purposes. And a GPAI model with systemic risk is a subset defined by Article 51: a model is so classified where it has high impact capabilities, with Article 51(2) presuming high impact capabilities where the cumulative compute used for training exceeds 10^25 floating point operations. Obligations for all GPAI model providers are set out in Article 53, covering technical documentation, information for downstream providers, a copyright compliance policy, and a public summary of training content. Article 55 adds obligations for systemic-risk models.

**Conclusion:** A GPAI model is defined by Article 3(63): a model displaying significant generality, capable of competently performing a wide range of distinct tasks and integrable into downstream systems, excluding pre-market research and prototyping models.""",

"q06": """**Issue:** Who is defined as a 'provider' under the EU AI Act?

**Rule:** Article 3(3) defines a provider as a natural or legal person, public authority, agency or other body that develops an AI system or a general-purpose AI model, or that has an AI system or a general-purpose AI model developed, and places it on the market or puts the AI system into service under its own name or trademark, whether for payment or free of charge.

**Application:** Two elements matter. First, the definition turns on placing on the market or putting into service under one's own name or trademark, not on who wrote the code: an organisation that commissions development from a third party is the provider. Second, it is free of any payment requirement, so distributing a system at no charge does not avoid provider status. The provider is distinct from the deployer (Article 3(4)), the importer (Article 3(6)), the distributor (Article 3(7)) and the authorised representative (Article 3(5)). Article 25 sets out the circumstances in which a distributor, importer, deployer or other third party is deemed to become the provider of a high-risk AI system: putting its name or trademark on a system already placed on the market, making a substantial modification to it, or modifying its intended purpose such that it becomes high-risk. Provider obligations for high-risk systems are consolidated in Article 16.

**Conclusion:** A provider under Article 3(3) is the person or body that develops, or has developed, an AI system or GPAI model and places it on the market or into service under its own name or trademark, whether or not for payment.""",

"q07": """**Issue:** Who is defined as a 'deployer' under the EU AI Act?

**Rule:** Article 3(4) defines a deployer as a natural or legal person, public authority, agency or other body using an AI system under its authority, except where the AI system is used in the course of a personal non-professional activity.

**Application:** The definition is not limited to high-risk systems, and it does not depend on the processing of personal data. Its two operative limbs are use under one's own authority and a professional context; the personal non-professional use carve-out keeps private individuals outside the regime. Deployer obligations for high-risk AI systems are set out in Article 26 and include using the system in accordance with the instructions for use, assigning human oversight to natural persons with the necessary competence, training and authority, ensuring input data is relevant and sufficiently representative so far as the deployer controls it, monitoring operation and suspending use where a risk is identified, keeping automatically generated logs for at least six months, and informing affected persons where the system is used in decisions concerning them. Article 27 requires a fundamental rights impact assessment from certain deployers. Under Article 25, a deployer that puts its own name or trademark on a high-risk system, substantially modifies it, or changes its intended purpose so that it becomes high-risk is treated as a provider.

**Conclusion:** A deployer under Article 3(4) is any person or body using an AI system under its own authority in a professional capacity, with obligations set out principally in Articles 26 and 27.""",

"q08": """**Issue:** What are the transparency obligations for AI systems that interact with natural persons under Article 50?

**Rule:** Article 50(1) requires providers to ensure that AI systems intended to interact directly with natural persons are designed and developed so that the persons concerned are informed that they are interacting with an AI system, unless this is obvious from the point of view of a reasonably well-informed, observant and circumspect natural person taking into account the circumstances and context of use. An exception applies to systems authorised by law to detect, prevent, investigate or prosecute criminal offences, subject to safeguards for the rights of third parties.

**Application:** Article 50 imposes four further duties. Article 50(2) requires providers of systems generating synthetic audio, image, video or text, including general-purpose AI systems, to mark outputs in a machine-readable format detectable as artificially generated or manipulated, using solutions that are effective, interoperable, robust and reliable as far as technically feasible. Article 50(3) requires deployers of emotion recognition or biometric categorisation systems to inform the persons exposed. Article 50(4) requires deployers to disclose deep fakes, with a lighter disclosure where the content is part of an evidently artistic, creative, satirical or fictional work, and to disclose AI-generated or manipulated text published to inform the public on matters of public interest, unless the content underwent human review with a natural or legal person holding editorial responsibility. Article 50(5) requires that the information be given clearly and distinguishably at the latest at the time of the first interaction or exposure, and conform to accessibility requirements. Article 50(6) preserves the Chapter III requirements.

**Conclusion:** Article 50 requires disclosure that a person is interacting with an AI system, machine-readable marking of synthetic content, notification for emotion recognition and biometric categorisation, and disclosure of deep fakes and AI-generated public-interest text, given clearly at first interaction.""",

"q09": """**Issue:** What is a conformity assessment under the EU AI Act?

**Rule:** Conformity assessment for high-risk AI systems is governed by Article 43. It is the procedure by which a provider demonstrates, before placing a high-risk AI system on the market or putting it into service, that the system meets the requirements of Chapter III Section 2 (Articles 8-15). The route depends on the type of system. Under Article 43(1), for systems listed in Annex III point 1 (biometrics), where the provider has applied harmonised standards or, where they exist, common specifications, it may choose between the internal control procedure in Annex VI and the procedure based on assessment of the quality management system and technical documentation with the involvement of a notified body in Annex VII; where such standards have not been applied or only partly applied, the Annex VII procedure is mandatory. Under Article 43(2), for systems listed in Annex III points 2 to 8, the provider follows the internal control procedure in Annex VI, without a notified body.

**Application:** Under Article 43(3), where the high-risk system relates to a product covered by the Annex I Section A harmonisation legislation, the provider follows the conformity assessment procedure required under that legislation, with the AI Act requirements forming part of that assessment. Article 43(4) requires a fresh assessment where the system undergoes a substantial modification. The assessment is followed by the EU declaration of conformity under Article 47 and the CE marking under Article 48, and is evidenced by the technical documentation required by Article 11 and Annex IV.

**Conclusion:** A conformity assessment under Article 43 is the pre-market demonstration that a high-risk AI system satisfies Articles 8-15, carried out by internal control under Annex VI or with a notified body under Annex VII depending on the system's classification.""",

"q10": """**Issue:** What are the record-keeping requirements under Article 12?

**Rule:** Article 12(1) requires that high-risk AI systems technically allow for the automatic recording of events, referred to as logs, over the lifetime of the system. Article 12(2) provides that, to ensure a level of traceability appropriate to the intended purpose, logging capabilities must enable the recording of events relevant to identifying situations that may result in the system presenting a risk within the meaning of Article 79(1) or in a substantial modification, to facilitating the post-market monitoring referred to in Article 72, and to monitoring the operation of the system as referred to in Article 26(5).

**Application:** Article 12(3) imposes a specific minimum for the remote biometric identification systems referred to in Annex III point 1(a): the logs must record the period of each use, with start and end date and time; the reference database against which input data has been checked; the input data for which the search led to a match; and the identification of the natural persons involved in verifying the results under Article 14(5). Retention is dealt with elsewhere. Article 19 requires providers to keep the logs automatically generated by their high-risk AI systems, so far as those logs are under their control, for a period appropriate to the intended purpose and at least six months. Article 26(6) places a corresponding obligation on deployers to keep logs for at least six months. Record-keeping is therefore a design requirement on the system under Article 12 and a retention obligation on providers and deployers under Articles 19 and 26(6).

**Conclusion:** Article 12 requires high-risk AI systems to support automatic event logging across their lifetime, sufficient for traceability, risk identification, post-market monitoring and operational oversight, with enhanced content requirements for remote biometric identification systems.""",

"q11": """**Issue:** What are the compliance obligations for providers of high-risk AI systems and how do they compare to deployers' obligations?

**Rule:** Provider obligations are consolidated in Article 16. Providers must ensure their high-risk AI systems comply with the requirements of Chapter III Section 2 (Articles 8-15), covering the risk management system (Article 9), data and data governance (Article 10), technical documentation (Article 11), record-keeping (Article 12), transparency and instructions for use (Article 13), human oversight (Article 14) and accuracy, robustness and cybersecurity (Article 15). They must indicate their name and address on the system, operate a quality management system (Article 17), keep documentation (Article 18) and automatically generated logs (Article 19), ensure the system undergoes conformity assessment before placing on the market (Article 43), draw up the EU declaration of conformity (Article 47), affix the CE marking (Article 48), register in the EU database (Article 49), take corrective action where a system is non-conforming (Article 20), and demonstrate conformity on a reasoned request (Article 21). Providers also carry post-market monitoring (Article 72) and serious incident reporting (Article 73).

**Application:** Deployer obligations sit in Article 26 and are operational rather than product-focused: using the system in accordance with the instructions for use, assigning human oversight to competent and trained natural persons with the necessary authority and support, ensuring input data is relevant and sufficiently representative so far as they control it, monitoring operation and suspending use and informing the provider and market surveillance authority where a risk is identified, keeping logs for at least six months, informing workers' representatives before workplace deployment, and informing natural persons subject to decisions taken with the system. Article 27 adds a fundamental rights impact assessment for public bodies, private entities providing public services and deployers of certain Annex III point 5 systems.

**Conclusion:** Provider obligations are ex ante and attach to the product through design, documentation, conformity assessment and marking; deployer obligations are ongoing and attach to use through oversight, input control, monitoring and notification. Article 25 converts a deployer into a provider where it rebrands, substantially modifies, or changes the intended purpose of a high-risk system.""",

"q12": """**Issue:** What is the timeline for when different provisions of the EU AI Act enter into force, specifically Article 5 and Chapter V?

**Rule:** Article 113 governs entry into force and application. The Regulation entered into force on the twentieth day following its publication in the Official Journal, which was on 12 July 2024, giving an entry into force of 1 August 2024. Article 113 then provides that the Regulation applies generally from 2 August 2026, subject to three exceptions: under point (a), Chapters I and II apply from 2 February 2025; under point (b), Chapter III Section 4, Chapter V, Chapter VII and Chapter XII, together with Article 78, apply from 2 August 2025, with the exception of Article 101; and under point (c), Article 6(1) and the corresponding obligations apply from 2 August 2027.

**Application:** Article 5 sits in Chapter II, so the prohibitions on unacceptable-risk practices took effect on 2 February 2025 under Article 113(a), together with the Chapter I general provisions including the Article 4 AI literacy obligation. Chapter V, which contains the obligations for providers of general-purpose AI models in Articles 53 to 55, took effect on 2 August 2025 under Article 113(b), alongside Chapter VII on governance and Chapter XII on penalties. Article 101, which sets fines for GPAI model providers, is expressly carved out of that earlier date. The general body of high-risk obligations applies from 2 August 2026, and the Article 6(1) route covering AI systems that are safety components of regulated products is deferred to 2 August 2027.

**Conclusion:** Article 5 has applied since 2 February 2025 as part of Chapter II; Chapter V has applied since 2 August 2025, excluding Article 101; the general application date is 2 August 2026, with Article 6(1) deferred to 2 August 2027.""",

"q13": """**Issue:** What are the risk classification tiers in the EU AI Act and what obligations correspond to each tier?

**Rule:** The Act does not use the phrase "tiers", but it operates a risk-based structure with four levels of obligation for AI systems, plus a separate regime for general-purpose AI models. Unacceptable risk is addressed by Article 5, which prohibits eight categories of practice outright. High risk is defined by Article 6, covering systems that are safety components of, or are, products under the Annex I harmonisation legislation requiring third-party conformity assessment, and the systems listed in Annex III, subject to the Article 6(3) derogation. Transparency risk is addressed by Article 50, which imposes disclosure duties on certain systems irrespective of whether they are high-risk. Minimal risk attracts no mandatory obligations, with Article 95 encouraging voluntary codes of conduct.

**Application:** High-risk classification carries the heaviest load: the requirements of Chapter III Section 2 (Articles 8-15), provider obligations under Article 16, deployer obligations under Article 26, a fundamental rights impact assessment under Article 27 where applicable, conformity assessment under Article 43, declaration of conformity under Article 47, CE marking under Article 48, registration under Article 49, and post-market monitoring and incident reporting under Articles 72 and 73. Transparency-risk systems owe only the Article 50 disclosure duties. Cutting across all levels, Article 4 requires providers and deployers to take measures to ensure a sufficient level of AI literacy among their staff. General-purpose AI models are regulated on a separate axis under Chapter V, with Article 53 obligations for all such models and Article 55 obligations added for models classified under Article 51 as presenting systemic risk.

**Conclusion:** The Act layers obligations across prohibited practices under Article 5, high-risk systems under Article 6 with Articles 8-15 and 16, 26, 43, 47, 48 and 49, transparency-risk systems under Article 50, and minimal-risk systems with no mandatory duties, alongside the separate GPAI regime in Chapter V.""",

"q14": """**Issue:** What is the role and responsibilities of the European AI Board and how does it interact with national supervisory authorities?

**Rule:** Article 65 establishes the European Artificial Intelligence Board and sets out its structure. The Board is composed of one representative per Member State. The European Data Protection Supervisor participates as an observer, and the AI Office attends meetings without taking part in votes. Article 65 also provides for the Board to set up standing or temporary sub-groups and to invite national and Union authorities, bodies and experts to attend as observers. Article 66 sets out the Board's tasks, which are to advise and assist the Commission and the Member States in order to facilitate the consistent and effective application of the Regulation, to coordinate among national competent authorities, to collect and share technical and regulatory expertise and best practices, to contribute to the harmonisation of administrative practices, and to issue recommendations and written opinions on relevant matters.

**Application:** The Board must be distinguished from the other governance bodies. Article 64 provides for the AI Office as a Commission function, which holds the exclusive supervisory and enforcement powers over general-purpose AI models under Article 88. Article 67 establishes an advisory forum for stakeholder input and Article 68 a scientific panel of independent experts. Enforcement on the ground rests with the Member States: Article 70 requires each Member State to designate at least one notifying authority and at least one market surveillance authority as national competent authorities, and market surveillance under Chapter IX, in particular Article 74, is exercised by those authorities. The Board is therefore a coordinating and advisory body rather than an enforcement body, providing the mechanism through which national authorities and the Commission align their practice.

**Conclusion:** The Board is established by Article 65 with one representative per Member State and carries the advisory and coordination tasks in Article 66. It harmonises the practice of the national competent authorities designated under Article 70, which retain enforcement powers, while the AI Office under Article 64 enforces the GPAI regime.""",

"q15": """**Issue:** How do the requirements for post-market monitoring and market surveillance under the EU AI Act for high-risk AI systems compare?

**Rule:** The two are distinct mechanisms with different duty-holders. Post-market monitoring is a provider obligation under Article 72, which requires providers to establish and document a post-market monitoring system proportionate to the nature of the AI technologies and the risks of the high-risk AI system. That system must actively and systematically collect, document and analyse relevant data on the performance of the system throughout its lifetime, whether provided by deployers or collected through other sources, and must allow the provider to evaluate continuous compliance with the requirements in Chapter III Section 2. It is based on a post-market monitoring plan forming part of the technical documentation referred to in Annex IV. Market surveillance is an authority function under Chapter IX, in particular Article 74, exercised by the market surveillance authorities designated by Member States under Article 70, applying Regulation (EU) 2019/1020.

**Application:** The bridge between them is Article 73, which requires providers to report serious incidents to the market surveillance authorities of the Member State where the incident occurred, immediately after establishing a causal link and in any event not later than fifteen days after becoming aware, with shortened deadlines for widespread infringements and serious disruption to critical infrastructure and for the death of a person. Market surveillance authorities are given access rights, including to documentation and, where necessary and subject to conditions, to training, validation and testing datasets and source code. Article 79 sets out the procedure for AI systems presenting a risk at national level, Article 80 the procedure for systems the provider classified as not high-risk under Article 6(3), and Article 82 the position for compliant systems that nonetheless present a risk.

**Conclusion:** Post-market monitoring under Article 72 is the provider's own continuous surveillance of its system in the field; market surveillance under Article 74 is public enforcement by national authorities. They connect through the Article 73 serious incident reporting duty and the authorities' investigatory powers.""",

"q16": """**Issue:** How does the EU AI Act address general-purpose AI (GPAI) models with systemic risks compared to regular GPAI models?

**Rule:** Chapter V regulates GPAI models on two levels. Article 53 sets the baseline for all providers of GPAI models: drawing up and keeping up to date technical documentation of the model including its training and testing process, as specified in Annex XI; making information available to downstream providers who integrate the model, as specified in Annex XII; putting in place a policy to comply with Union law on copyright and related rights, including identifying and respecting reservations of rights under Article 4(3) of Directive (EU) 2019/790; and making publicly available a sufficiently detailed summary of the content used for training, according to a template provided by the AI Office. Article 53(2) exempts models released under a free and open-source licence from the first two of those duties, but that exemption does not extend to models with systemic risk.

**Application:** Article 51 governs classification. A GPAI model is classified as presenting systemic risk where it has high impact capabilities evaluated on the basis of appropriate technical tools and methodologies, or where the Commission so decides, on its own initiative or following a qualified alert from the scientific panel, having regard to the criteria in Annex XIII. Article 51(2) creates a presumption of high impact capabilities where the cumulative amount of computation used for training exceeds 10^25 floating point operations. Article 52 requires the provider to notify the Commission without delay and within two weeks of meeting or expecting to meet that threshold, permits the provider to argue that the model exceptionally does not present systemic risk, and requires the Commission to maintain a public list. Article 55 then adds obligations that apply only to such models: performing model evaluation including adversarial testing to identify and mitigate systemic risk, assessing and mitigating possible systemic risks at Union level, tracking, documenting and reporting serious incidents and corrective measures to the AI Office without undue delay, and ensuring an adequate level of cybersecurity protection for the model and its physical infrastructure.

**Conclusion:** All GPAI models attract the Article 53 documentation, downstream information, copyright and training-summary duties. Models classified under Article 51 as presenting systemic risk additionally attract the Article 55 evaluation, risk mitigation, incident reporting and cybersecurity duties, lose the open-source exemption, and fall under Commission enforcement with fines under Article 101.""",

"q17": """**Issue:** What are the rights of deployers and individuals when AI systems make automated decisions, and how can they seek explanations?

**Rule:** Article 86 confers a right to explanation of individual decision-making. Any affected person subject to a decision taken by a deployer on the basis of the output of a high-risk AI system listed in Annex III, with the exception of the systems listed under point 2 of that Annex, which produces legal effects or similarly significantly affects that person in a way they consider to have an adverse impact on their health, safety or fundamental rights, has the right to obtain from the deployer clear and meaningful explanations of the role of the AI system in the decision-making procedure and of the main elements of the decision taken. Article 86(2) preserves exceptions and restrictions following from Union or national law, and Article 86(3) provides that the right applies only to the extent it is not otherwise provided for under Union law.

**Application:** Article 86 is supported by several adjacent provisions. Article 26(11) requires deployers of Annex III high-risk systems that make or assist in making decisions about natural persons to inform those persons that they are subject to the use of the system. Article 50 requires disclosure that a person is interacting with an AI system and disclosure of deep fakes and certain AI-generated text. Article 85 gives any natural or legal person with grounds to consider that the Regulation has been infringed the right to lodge a complaint with the relevant market surveillance authority. Article 27 requires certain deployers to carry out a fundamental rights impact assessment before putting a high-risk system into use. On the deployer side, the corresponding duty is the human oversight obligation in Article 26(2), which requires oversight to be assigned to natural persons with the necessary competence, training and authority. Under Article 113, Article 86 applies from 2 August 2026.

**Conclusion:** Individuals affected by decisions made on the basis of Annex III high-risk systems have a right under Article 86 to clear and meaningful explanations from the deployer, reinforced by the notification duty in Article 26(11) and the right to complain to a market surveillance authority under Article 85.""",

"q18": """**Issue:** What is the role of the EU AI Office, its structure, and how does it enforce compliance for general-purpose AI models?

**Rule:** Article 3(47) defines the AI Office as the Commission's function of contributing to the implementation, monitoring and supervision of AI systems and general-purpose AI models, and to AI governance. Article 64 provides that the Commission shall develop Union expertise and capabilities in the field of AI through the AI Office, and that Member States shall facilitate the tasks entrusted to it. Structurally, the AI Office is a function within the Commission rather than an independent agency, and it is distinct from the European Artificial Intelligence Board established by Article 65, the advisory forum under Article 67 and the scientific panel of independent experts under Article 68.

**Application:** Enforcement for general-purpose AI models is centralised. Article 88 gives the Commission exclusive powers to supervise and enforce Chapter V, and provides that those powers are entrusted to the AI Office, without prejudice to the powers of national authorities in respect of AI systems. Article 89 provides for monitoring actions, including the power to take the steps necessary to monitor the effective implementation of Chapter V. Article 90 allows the scientific panel to issue qualified alerts where it has reason to suspect a model presents systemic risk. Article 91 gives the power to request information from providers, Article 92 the power to conduct evaluations of models, including through independent experts, and Article 93 the power to request that providers take measures, including implementing mitigations, restricting availability, withdrawing or recalling a model. Article 94 preserves the procedural rights of economic operators. Financial penalties for GPAI providers are set by Article 101 at up to 3 % of total worldwide annual turnover or EUR 15 000 000, whichever is higher. The AI Office also facilitates codes of practice under Article 56 and provides the training-content summary template referred to in Article 53(1)(d).

**Conclusion:** The AI Office is a Commission function under Article 64, distinct from the Board under Article 65. It holds the Commission's exclusive Chapter V enforcement powers through Article 88, exercised via Articles 89 to 93 and backed by fines under Article 101.""",

"q19": """**Issue:** What documentation must be prepared for high-risk AI systems under Article 11 and how does it relate to the conformity assessment?

**Rule:** Article 11(1) requires the technical documentation of a high-risk AI system to be drawn up before that system is placed on the market or put into service, and to be kept up to date. It must be drawn up in such a way as to demonstrate that the system complies with the requirements of Chapter III Section 2, and to provide national competent authorities and notified bodies with the necessary information, in a clear and comprehensive form, to assess that compliance. It must contain, at a minimum, the elements set out in Annex IV. Providers that are SMEs, including start-ups, may provide those elements in a simplified manner, using a simplified form to be established by the Commission. Article 11(2) provides that where a high-risk system related to a product covered by the Annex I Section A legislation is placed on the market, a single set of technical documentation is drawn up covering both the Annex IV information and the information required under those legal acts.

**Application:** Annex IV specifies the content, including a general description of the system, a detailed description of its elements and development process, information on monitoring, functioning and control, the appropriateness of the performance metrics, the risk management system required by Article 9, a description of relevant changes made through the lifecycle, the harmonised standards applied, a copy of the EU declaration of conformity, and a description of the post-market monitoring plan under Article 72. The relationship to conformity assessment is direct: the technical documentation is the evidence base on which the Article 43 assessment is conducted. Under the Annex VI internal control route the provider itself verifies that the documentation demonstrates compliance; under the Annex VII route the notified body assesses that documentation. Article 18 requires the provider to keep the documentation at the disposal of national competent authorities for ten years after the system is placed on the market or put into service, and Article 21 requires it to be supplied on a reasoned request.

**Conclusion:** Article 11 requires Annex IV technical documentation, drawn up before placing on the market and kept current, demonstrating compliance with Articles 8-15. It is the substantive input to the Article 43 conformity assessment and must be retained for ten years under Article 18.""",

"q20": """**Issue:** What is the concept of 'human oversight' under Article 14 and how must it be implemented in high-risk AI systems?

**Rule:** Article 14(1) requires high-risk AI systems to be designed and developed in such a way, including with appropriate human-machine interface tools, that they can be effectively overseen by natural persons during the period in which they are in use. Article 14(2) provides that human oversight shall aim to prevent or minimise the risks to health, safety or fundamental rights that may emerge when the system is used in accordance with its intended purpose or under conditions of reasonably foreseeable misuse, in particular where such risks persist despite the application of the other requirements in Chapter III Section 2. Article 14(3) requires the oversight measures to be commensurate with the risks, level of autonomy and context of use, and to be ensured through measures built into the system by the provider before it is placed on the market where technically feasible, or measures identified by the provider as appropriate for the deployer to implement, or both.

**Application:** Article 14(4) specifies the capabilities the system must enable in the natural persons assigned to oversight, as appropriate and proportionate: to properly understand the relevant capacities and limitations of the system and monitor its operation so that anomalies, dysfunctions and unexpected performance can be detected and addressed; to remain aware of the possible tendency to automatically rely or over-rely on the output, known as automation bias; to correctly interpret the output; to decide, in any particular situation, not to use the system or otherwise to disregard, override or reverse its output; and to intervene in its operation or interrupt it through a stop button or similar procedure allowing the system to come to a halt in a safe state. Article 14(5) adds a specific safeguard for the remote biometric identification systems in Annex III point 1(a): no action or decision may be taken on the basis of an identification unless it has been separately verified and confirmed by at least two natural persons with the necessary competence, training and authority, though that requirement does not apply where Union or national law considers it disproportionate in the law enforcement, migration, border control or asylum context. On the deployer side, Article 26(2) requires oversight to be assigned to natural persons with the necessary competence, training and authority, together with the necessary support.

**Conclusion:** Human oversight under Article 14 is a design obligation on the provider to make effective oversight possible, coupled with an operational obligation on the deployer under Article 26(2) to resource it. It requires understanding of the system's limits, awareness of automation bias, correct interpretation of output, authority to disregard or override, and the ability to halt the system safely.""",
}

# --- q21-q30: news-seeking, no news source exists ----------------------------
# Rewritten to reward the correct behaviour under the benchmark's actual
# configuration. Left needs_review=True: reframing these is a design decision.
ROUTING = {

"q21": """**Issue:** What are the latest news updates regarding the establishment and active operations of the EU AI Office?

**Rule:** Article 3(47) defines the AI Office as the Commission's function of contributing to the implementation, monitoring and supervision of AI systems and general-purpose AI models, and to AI governance. Article 64 provides that the Commission shall develop Union expertise and capabilities in the field of AI through the AI Office, and that Member States shall facilitate its tasks. Article 88 entrusts the Commission's exclusive powers to supervise and enforce Chapter V to the AI Office, exercised through the monitoring, information, evaluation and measures powers in Articles 89 to 93. Under Article 113(b), Chapter V and the governance provisions in Chapter VII have applied since 2 August 2025.

**Application:** The question asks for current news. The available corpus is the text of Regulation (EU) 2024/1689 as published in the Official Journal; it contains no news reporting, and no live news or web source is enabled for this query. The statutory basis, mandate and enforcement powers of the AI Office can therefore be stated with confidence, but its present staffing, published guidance, and day-to-day operational activity cannot be established from the available sources.

**Conclusion:** The AI Office is established as a Commission function under Article 64 and holds the Chapter V enforcement mandate under Article 88, in force since 2 August 2025. Current operational developments cannot be confirmed from the available sources and should be verified against the Commission's official publications.""",

"q22": """**Issue:** How does the EU AI Act regulate general-purpose AI (GPAI) models, and what recent announcements or updates happened this week?

**Rule:** Chapter V governs GPAI models. Article 53 requires all providers to maintain technical documentation of the model in accordance with Annex XI, to provide information to downstream providers in accordance with Annex XII, to put in place a policy to comply with Union copyright law including reservations of rights under Article 4(3) of Directive (EU) 2019/790, and to publish a sufficiently detailed summary of training content using the AI Office template. Article 51 classifies a model as presenting systemic risk where it has high impact capabilities, presumed under Article 51(2) where training compute exceeds 10^25 floating point operations. Article 55 adds evaluation, adversarial testing, risk mitigation, incident reporting and cybersecurity obligations for such models. Article 56 provides for codes of practice, and Article 101 sets fines of up to 3 % of worldwide annual turnover or EUR 15 000 000, whichever is higher.

**Application:** The second limb of the question asks for developments from the current week. The available corpus is the statutory text of Regulation (EU) 2024/1689 only, with no news source enabled, and it carries no date-stamped reporting. Recent announcements therefore cannot be identified or dated from the available sources.

**Conclusion:** The GPAI regime rests on Articles 51 to 55, with codes of practice under Article 56 and fines under Article 101, applicable since 2 August 2025 under Article 113(b). No recent announcements can be confirmed from the available sources.""",

"q23": """**Issue:** What recent updates or announcements were made regarding the EU-US AI safety cooperation agreement?

**Rule:** Regulation (EU) 2024/1689 does not establish or govern any bilateral EU-US AI safety cooperation agreement. The Regulation's territorial reach is set by Article 2, which applies it to providers placing AI systems on the Union market irrespective of where they are established, and to providers and deployers established in a third country where the output produced by the system is used in the Union. International alignment is addressed at the level of cooperation with third countries and international organisations rather than through any named bilateral instrument.

**Application:** The question asks for recent updates on a specific instrument that does not appear in the Act. The available corpus is the statutory text only, with no news source enabled, so neither the existence nor the current status of such an agreement can be established from it. Answering the question as posed would require sources outside the corpus.

**Conclusion:** No EU-US AI safety cooperation agreement is established by the Regulation, and no recent updates on any such agreement can be confirmed from the available sources. The Act's extraterritorial reach derives from Article 2. Current diplomatic developments should be verified against official EU and US government sources.""",

"q24": """**Issue:** What are the latest news developments regarding Member States appointing their national market surveillance authorities?

**Rule:** Article 70 requires each Member State to establish or designate at least one notifying authority and at least one market surveillance authority as national competent authorities for the purpose of the Regulation, to ensure they are provided with adequate technical, financial and human resources and infrastructure, and to communicate their identity to the Commission. Market surveillance is then exercised under Chapter IX, in particular Article 74, which applies Regulation (EU) 2019/1020 and confers powers including access to documentation and, where necessary and subject to conditions, to training, validation and testing datasets and source code. Article 79 sets out the procedure for AI systems presenting a risk at national level.

**Application:** The question asks which Member States have made appointments and when. That is a matter of current administrative fact rather than statutory content. The available corpus is the text of the Regulation only, with no news source enabled, so the state of national designations cannot be determined from it.

**Conclusion:** The obligation to designate national competent authorities, including at least one market surveillance authority, arises under Article 70. The current status of Member State appointments cannot be confirmed from the available sources and should be verified against the Commission's published list of notified national competent authorities.""",

"q25": """**Issue:** Are there any recent announcements or guidelines released by the EU AI Office regarding general-purpose AI model classification?

**Rule:** Classification of general-purpose AI models is governed by Article 51, under which a model is classified as presenting systemic risk where it has high impact capabilities evaluated using appropriate technical tools and methodologies, or where the Commission so decides having regard to the criteria in Annex XIII. Article 51(2) presumes high impact capabilities where cumulative training compute exceeds 10^25 floating point operations. Article 52 sets the notification procedure: the provider must notify the Commission without delay and within two weeks of meeting or expecting to meet the threshold, may argue that the model exceptionally does not present systemic risk, and the Commission maintains a public list of such models. Article 56 provides for codes of practice facilitated by the AI Office, and Article 53(1)(d) refers to a training-content summary template provided by the AI Office.

**Application:** The question asks about guidance published by the AI Office. The Act anticipates such instruments, in the form of the Article 56 codes of practice and the Article 53(1)(d) template, but the corpus available here is the statutory text only, with no news source enabled. Whether particular guidance has been issued, and its content, cannot be established from it.

**Conclusion:** The classification framework rests on Articles 51 and 52, with Annex XIII criteria, and the Act envisages AI Office instruments under Articles 56 and 53(1)(d). Whether specific guidance has been released cannot be confirmed from the available sources.""",

"q26": """**Issue:** What recent news highlights the global impact of the EU AI Act on US technology companies?

**Rule:** The relevant statutory basis is Article 2, which sets the scope. The Regulation applies to providers placing AI systems on the Union market or putting them into service in the Union, irrespective of whether those providers are established in the Union or in a third country; to deployers established or located in the Union; and to providers and deployers established or located in a third country where the output produced by the AI system is used in the Union. It also applies to importers and distributors, to product manufacturers placing systems on the market together with their product, and to authorised representatives of providers not established in the Union. Article 22 requires providers established in third countries to appoint an authorised representative in the Union before making a high-risk AI system available.

**Application:** The extraterritorial reach in Article 2 is what brings non-EU providers, including US companies, within the regime, and Article 22 gives that reach a concrete compliance mechanism. The question, however, asks for recent news about commercial and political impact. The available corpus is the statutory text only, with no news source enabled, so specific corporate responses, enforcement actions or commentary cannot be identified from it.

**Conclusion:** US and other non-EU providers fall within scope through Article 2 where they place systems on the Union market or where system output is used in the Union, with the Article 22 authorised representative requirement for high-risk systems. Recent news on the commercial impact cannot be confirmed from the available sources.""",

"q27": """**Issue:** What are the latest news updates regarding the conformity assessment bodies for high-risk AI systems?

**Rule:** Conformity assessment bodies, referred to as notified bodies once designated, are governed by Chapter III Section 4, which under Article 113(b) has applied since 2 August 2025. Member States designate notifying authorities under Article 70 responsible for the procedures for assessment, designation and notification of conformity assessment bodies and for their monitoring. Articles 29 to 39 govern the application for notification, the requirements relating to notified bodies, their operational obligations, their subsidiaries and subcontractors, and challenges to their competence, with Article 39 addressing conformity assessment bodies established under the law of a third country. The assessment procedures themselves are set out in Article 43, which routes systems either to internal control under Annex VI or to a procedure involving a notified body under Annex VII.

**Application:** The question asks for current news about which bodies have been designated and their readiness. That is a matter of present administrative fact. The available corpus is the statutory text only, with no news source enabled, so the current state of designations cannot be established from it.

**Conclusion:** The framework for conformity assessment bodies rests on Chapter III Section 4, Articles 29 to 39, with assessment procedures under Article 43, in application since 2 August 2025. The current status of notified body designations cannot be confirmed from the available sources.""",

"q28": """**Issue:** What news updates exist regarding the enforcement of the Article 5 prohibitions which entered into force recently?

**Rule:** Article 5 prohibits eight categories of AI practice, including manipulative or deceptive techniques materially distorting behaviour, exploitation of vulnerabilities, social scoring, predicting criminal offences based solely on profiling or personality traits, untargeted scraping of facial images to build facial recognition databases, emotion inference in the workplace and education, biometric categorisation to infer sensitive attributes, and real-time remote biometric identification in publicly accessible spaces for law enforcement subject to narrow exceptions. Under Article 113(a), Chapters I and II, which contain Article 5, have applied since 2 February 2025. Penalties for breach are set by Article 99(3) at up to EUR 35 000 000 or 7 % of total worldwide annual turnover, whichever is higher, and Chapter XII containing Article 99 has applied since 2 August 2025 under Article 113(b).

**Application:** The question asks for news about enforcement activity under these provisions. The available corpus is the statutory text only, with no news source enabled, so specific investigations, decisions or fines cannot be identified or dated from it.

**Conclusion:** The Article 5 prohibitions have applied since 2 February 2025, with the Article 99(3) penalty regime available since 2 August 2025. No specific enforcement actions can be confirmed from the available sources and should be verified against national market surveillance authority and Commission publications.""",

"q29": """**Issue:** Are there any recent articles or news on the penalties imposed on companies under the EU AI Act?

**Rule:** Penalties are governed by Article 99. Article 99(1) requires Member States to lay down rules on penalties that are effective, proportionate and dissuasive. Article 99(3) sets fines for non-compliance with the Article 5 prohibitions at up to EUR 35 000 000 or, for an undertaking, up to 7 % of total worldwide annual turnover for the preceding financial year, whichever is higher. Article 99(4) sets fines for other specified infringements at up to EUR 15 000 000 or 3 %, whichever is higher. Article 99(5) sets fines for supplying incorrect, incomplete or misleading information to notified bodies or national competent authorities at up to EUR 7 500 000 or 1 %, whichever is higher. Article 99(6) caps each fine for SMEs including start-ups at the lower of the applicable percentage or fixed amount. Article 101 separately provides fines for providers of general-purpose AI models of up to 3 % or EUR 15 000 000, whichever is higher.

**Application:** Chapter XII, containing Article 99, has applied since 2 August 2025 under Article 113(b), and Article 101 is expressly excluded from that earlier date. The question asks whether penalties have in fact been imposed on identified companies. The available corpus is the statutory text only, with no news source enabled, so no enforcement outcome can be identified from it.

**Conclusion:** The penalty ceilings are set by Article 99 at EUR 35 000 000 or 7 %, EUR 15 000 000 or 3 %, and EUR 7 500 000 or 1 % depending on the infringement, with Article 101 governing GPAI model providers. Whether any fines have been imposed cannot be confirmed from the available sources.""",

"q30": """**Issue:** What is the latest news about standardisation requests for AI systems under the EU AI Act?

**Rule:** Standardisation is governed by Article 40, which provides that high-risk AI systems or general-purpose AI models which are in conformity with harmonised standards, or parts thereof, the references of which have been published in the Official Journal in accordance with Regulation (EU) No 1025/2012, shall be presumed to be in conformity with the requirements of Chapter III Section 2 or, as applicable, the Chapter V obligations, to the extent those standards cover them. Article 41 provides for common specifications to be adopted by the Commission by implementing act where harmonised standards are insufficient, where the Commission's standardisation request has not been accepted, or where there are undue delays.

**Application:** The practical significance of Article 40 is that harmonised standards supply the presumption of conformity relied on in the Article 43 conformity assessment, and under Article 43(1) the availability of such standards determines whether a provider of an Annex III point 1 biometrics system may use the internal control route in Annex VI rather than the notified body route in Annex VII. The question asks about the current status of standardisation requests issued to the European standardisation organisations. The available corpus is the statutory text only, with no news source enabled, so the progress of any such request cannot be established from it.

**Conclusion:** Article 40 establishes the presumption of conformity from harmonised standards, with Article 41 providing for common specifications as a fallback. The current status of standardisation requests and published standards cannot be confirmed from the available sources.""",
}


def main() -> None:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    if not BACKUP.exists():
        shutil.copy2(DATASET, BACKUP)
        print(f"backed up drafts -> {BACKUP.name}")

    changed = 0
    for item in data:
        qid = item["id"]
        if qid in VERIFIED:
            item["gold"] = VERIFIED[qid]
            item["gold_status"] = "revised"
            item["needs_review"] = False
        elif qid in ROUTING:
            item["gold"] = ROUTING[qid]
            item["gold_status"] = "revised_no_news_source"
            item["needs_review"] = True
        else:
            raise SystemExit(f"unhandled id {qid}")
        item["gold_model"] = REVIEWER
        item["gold_revised_at"] = "2026-08-06"
        # The draft ids pointed into the stale 44-chunk news corpus and now
        # resolve to unrelated Act recitals; keeping them would misdescribe
        # provenance. Retrieval metrics stay off regardless (gold_doc_ids empty).
        item["retriever_ids_at_gold_draft_stale"] = item.pop(
            "retriever_ids_at_gold_draft", []
        )
        changed += 1

    DATASET.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"rewrote {changed} reference answers")
    print(f"  revised (verified, review cleared): {len(VERIFIED)}")
    print(f"  revised (news-seeking, still flagged): {len(ROUTING)}")


if __name__ == "__main__":
    main()
