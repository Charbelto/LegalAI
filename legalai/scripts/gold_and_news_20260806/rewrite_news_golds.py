"""Write the 10 routing golds against the frozen news snapshot.

Every fact below was read out of the chunks the retriever actually returns for
these queries (captured in routing_context.json), so the golds ask for nothing
the system cannot see. Statutory anchors stay verified against the Act text.

q12 also gets a hedged sentence. Its gold is statute-correct on Article 113, but
the corpus now carries reports that the Digital Omnibus defers the Annex III
high-risk date to 2 December 2027. Without the note a system that correctly
cites the deferral would be scored wrong against a gold that predates it.
"""

import json
from pathlib import Path

DATASET = Path(r"C:\Users\Charbel\Desktop\Legal AI\legalai\eval_dataset.json")
SNAPSHOT = "news snapshot 2026-08-06"

NEWS = {

"q21": """**Issue:** What are the latest news updates regarding the establishment and active operations of the EU AI Office?

**Rule:** Article 64 provides that the Commission shall develop Union expertise and capabilities in the field of AI through the AI Office, and Article 3(47) defines it as a Commission function. Article 88 entrusts to the AI Office the Commission's exclusive powers to supervise and enforce Chapter V on general-purpose AI models, exercised through the powers in Articles 89 to 93.

**Application:** On the available reporting, the AI Office is established and operational, and the Act has moved into an implementation phase in which compliance is shaped not only by the original text but by a wave of delegated and implementing acts together with AI Office guidance. Two developments are prominent. First, on 2 August 2026 the AI Office and the national authorities gained full powers to enforce and sanction breaches of obligations already in force, including in respect of general-purpose AI models, so the Office moved from a standard-setting posture to an enforcing one. Second, the AI Office is reported to define the priorities of the Union's international engagement on AI, to represent the EU in international organisations and to develop strategic partnerships. Reporting also records that a code of practice on marking AI-generated content was finalised and attracted roughly 190 signatories. Separately, a provisional political agreement on the Digital Omnibus reached on 7 May 2026 is reported to defer the Annex III high-risk obligations to 2 December 2027, which changes the sequencing of the Office's high-risk workload without altering its GPAI mandate.

**Conclusion:** The AI Office is established under Article 64 and has held the Chapter V enforcement mandate under Article 88 since 2 August 2025. The significant recent development is that its enforcement and sanctioning powers became fully operational on 2 August 2026, alongside reported deferral of the high-risk regime to 2 December 2027.""",

"q22": """**Issue:** How does the EU AI Act regulate general-purpose AI (GPAI) models, and what recent announcements or updates have happened?

**Rule:** Chapter V governs GPAI models. Article 53 requires all providers to keep technical documentation in accordance with Annex XI, to supply information to downstream providers under Annex XII, to operate a policy for compliance with Union copyright law, and to publish a sufficiently detailed summary of training content on the AI Office template. Article 51 classifies a model as presenting systemic risk where it has high impact capabilities, presumed under Article 51(2) above 10^25 floating point operations of training compute, and Article 55 adds evaluation, adversarial testing, risk mitigation, incident reporting and cybersecurity duties for such models. Article 101 sets fines of up to EUR 15 000 000 or 3 % of worldwide annual turnover, whichever is higher.

**Application:** The substantive GPAI obligations in Articles 51 to 55 have applied since 2 August 2025. On the available reporting, the change that took effect on 2 August 2026 is enforcement rather than substance: the AI Office can now act on its enforcement and sanction powers over GPAI providers, including opening investigations and imposing administrative fines. Two soft-law instruments accompany the regime. The General-Purpose AI Code of Practice, a voluntary instrument prepared by independent experts in a multi-stakeholder process, was published on 10 July 2025 and addresses safety, transparency and copyright. Draft Guidelines clarifying the GPAI provisions were published by the Commission on 18 July 2025, covering the definition and scope of GPAI models, lifecycle obligations, systemic risk criteria and notification duties, to be formally adopted once translated. Reporting also notes the Digital Omnibus provisional agreement of 7 May 2026, which defers the Annex III high-risk deadline to 2 December 2027 but does not move the GPAI dates.

**Conclusion:** GPAI models are regulated under Articles 51 to 55, in force since 2 August 2025, supported by the Code of Practice of 10 July 2025 and the draft Guidelines of 18 July 2025. The recent development is that AI Office enforcement and fining powers over GPAI providers became operational on 2 August 2026.""",

"q23": """**Issue:** What recent updates or announcements were made regarding EU-US AI safety cooperation?

**Rule:** Regulation (EU) 2024/1689 does not itself establish any bilateral EU-US instrument. Its reach over non-EU actors comes from Article 2, which applies the Regulation to providers placing AI systems on the Union market irrespective of establishment, and to providers and deployers in third countries where the system's output is used in the Union. Coordination of the Union's external AI engagement is reported to sit with the AI Office, consistent with its Article 64 role.

**Application:** On the available reporting, the principal cooperation announcement was made on 5 April 2024, when EU and US leaders stated that the two sides would work together to evaluate artificial intelligence models. That commitment was made at the sixth meeting of the EU-US Trade and Technology Council, held in Leuven, Belgium, at which further cooperation on AI safety, 6G research and semiconductors was also announced. Reporting further records a technical dialogue between the US AI Safety Institute and the European AI Office. The sources describe a cooperation framework and a channel for technical exchange rather than a binding treaty instrument, and they do not record a formal EU-US AI safety agreement with legal effect.

**Conclusion:** The cooperation rests on the 5 April 2024 EU-US Trade and Technology Council commitment to jointly evaluate AI models, together with a technical dialogue between the US AI Safety Institute and the European AI Office. No binding bilateral agreement is recorded in the available sources, and the Act's own reach over US providers derives from Article 2.""",

"q24": """**Issue:** What are the latest news developments regarding Member States appointing their national market surveillance authorities?

**Rule:** Article 70 requires each Member State to establish or designate at least one notifying authority and at least one market surveillance authority, to ensure they have adequate technical, financial and human resources, to designate one market surveillance authority as the single point of contact, and to notify the Commission, which publishes the list. Article 70 also provides that for the AI systems in points 1, 6, 7 and 8 of Annex III, Member States shall designate as market surveillance authorities either the competent data protection supervisory authorities or another authority designated under the same conditions. The Act required this information to be made publicly available by 2 August 2025.

**Application:** On the available reporting, the designation deadline of 2 August 2025 has passed and Member States were expected to have designated and empowered their national competent authorities by that date, but national implementation remains uneven: published trackers of national implementation plans and designated authorities are described as works in progress that are still being updated as information becomes available, which indicates that designation is incomplete or inconsistently notified across the Union. The operative development is that on 2 August 2026 the enforcement machinery became operational and national market surveillance authorities gained their full powers, so the practical consequence of any remaining gaps in designation is now materially greater than it was during the preparatory period.

**Conclusion:** The obligation arises under Article 70 and fell due on 2 August 2025. Reporting indicates designation across Member States is still incomplete and tracked on a rolling basis, while national market surveillance powers became fully operational on 2 August 2026. The Commission's published list of single points of contact is the authoritative source for the current position.""",

"q25": """**Issue:** Are there any recent announcements or guidelines released by the EU AI Office regarding general-purpose AI model classification?

**Rule:** Classification is governed by Article 51, under which a GPAI model is classified as presenting systemic risk where it has high impact capabilities, with Article 51(2) presuming that where cumulative training compute exceeds 10^25 floating point operations, or where the Commission so decides having regard to the Annex XIII criteria. Article 52 requires the provider to notify the Commission without delay and within two weeks. Article 3(63) supplies the underlying definition. The Act contemplates AI Office instruments: Article 53(1)(d) refers to a training-content summary template provided by the AI Office, Article 56 provides for codes of practice it facilitates, and the scientific panel is tasked with advising on the classification of general-purpose AI models with systemic risk.

**Application:** On the available reporting, the Commission published draft Guidelines on 18 July 2025 clarifying key provisions applicable to GPAI models. Those Guidelines address the definition and scope of a GPAI model, related lifecycle obligations, the systemic risk criteria and the notification duties of providers, and they expand on the statutory definition. They were issued in draft and are to be formally adopted once translated into all EU languages, at which point they are described as carrying legal and operational relevance for providers. Separately, the General-Purpose AI Code of Practice was published on 10 July 2025 as a voluntary instrument covering safety, transparency and copyright.

**Conclusion:** Yes. The Commission published draft Guidelines on 18 July 2025 covering the definition and scope of GPAI models, systemic risk criteria and notification duties, pending formal adoption after translation, alongside the Code of Practice of 10 July 2025. The statutory framework they interpret is Articles 51 to 53 with Annex XIII.""",

"q26": """**Issue:** What recent news highlights the global impact of the EU AI Act on US technology companies?

**Rule:** The statutory hook is Article 2, which applies the Regulation to providers placing AI systems on the Union market or putting them into service in the Union irrespective of where they are established, and to providers and deployers established in a third country where the output produced by the system is used in the Union. Article 22 requires third-country providers of high-risk systems to appoint an authorised representative in the Union. Article 99 sets the penalty ceilings, the highest being EUR 35 000 000 or 7 % of total worldwide annual turnover.

**Application:** On the available reporting, the recurring theme is extraterritorial reach: a US company is in scope where its AI systems or their outputs touch EU users, regardless of whether it has a physical EU presence, and commentary gives worked examples such as a US recruitment platform screening CVs for a German office or a credit scoring system assessing EU residents' loan applications. This is characterised as a Brussels Effect, in which the size and influence of the EU market make its standards a global benchmark, with practical advice to build AI governance, revise contracts and maintain ongoing compliance monitoring. A second theme is a widely reported misreading of the Digital Omnibus: the political agreement of May 2026 deferred the headline high-risk deadline by sixteen months, to 2 December 2027, and commentary warns that US boards treating this as a general reprieve have misjudged it, because the deadlines that moved are not the ones most US companies were facing. Legal commentary published on 28 April 2026 advised US businesses operating high-risk systems to remain focused on the 2 August 2026 date.

**Conclusion:** The reported impact is that US companies fall within scope through Article 2 wherever their systems or outputs reach the Union, exposed to Article 99 penalties of up to EUR 35 000 000 or 7 %. The dominant recent story is the Digital Omnibus deferral of the high-risk deadline to 2 December 2027 and warnings that it is being misread as broader relief than it gives.""",

"q27": """**Issue:** What are the latest news updates regarding conformity assessment bodies for high-risk AI systems?

**Rule:** Conformity assessment is governed by Article 43, under which providers of high-risk AI systems must complete the applicable procedure before the system is placed on the market, either internal control under Annex VI or a procedure involving a notified body under Annex VII. Notified bodies verify conformity in accordance with Article 43, must be independent of the provider and of any operator with an economic interest in the system, and are required to avoid unnecessary burdens, taking due account of the provider's size and sector so as to minimise administrative cost for micro and small enterprises while preserving the required rigour. Article 43(4) requires a fresh assessment on substantial modification. Notifying authorities designated under Article 70 designate and supervise notified bodies.

**Application:** On the available reporting, conformity assessment is described as the formal route by which high-risk providers demonstrate compliance before placing a system on the EU market, leading to the CE marking declaration under Articles 47 and 48. The operationally significant development is timing rather than institutional change: the Digital Omnibus political agreement of May 2026 is reported to defer the Annex III high-risk obligations from 2 August 2026 to 2 December 2027, which correspondingly postpones the point at which most Annex III providers must have completed conformity assessment. The available sources do not report a list of designated notified bodies or their current capacity.

**Conclusion:** The framework rests on Article 43 with Annexes VI and VII, and on notified bodies designated and supervised by the Article 70 notifying authorities. The recent development is the reported deferral of Annex III high-risk obligations to 2 December 2027, postponing when conformity assessment binds most providers. The Commission's NANDO database is the authoritative source for current designations.""",

"q28": """**Issue:** What news updates exist regarding enforcement of the Article 5 prohibitions?

**Rule:** Article 5 prohibits eight categories of practice, including manipulative or deceptive techniques that materially distort behaviour, exploitation of vulnerabilities, social scoring, predicting criminal offences based solely on profiling or personality traits, untargeted scraping of facial images, emotion inference in the workplace and education, biometric categorisation to infer sensitive attributes, and real-time remote biometric identification in publicly accessible spaces for law enforcement subject to narrow exceptions. Under Article 113(a) the prohibitions have applied since 2 February 2025. Article 99(3) sets fines of up to EUR 35 000 000 or 7 % of total worldwide annual turnover for their breach.

**Application:** On the available reporting, the prohibitions have been in force since 2 February 2025 and the Article 99 penalty tiers have been enforceable since 2 August 2025. The material development is that on 2 August 2026, two years after entry into force, the Act's enforcement machinery became operational for obligations already in force, and the transparency obligations in Article 50 began to apply to a broad range of systems, providers and deployers. Enforcement capacity nonetheless depends on national designation, and trackers of national implementation plans and designated authorities are still described as works in progress. The available sources record the activation of enforcement powers but do not identify any specific investigation, decision or fine imposed under Article 5.

**Conclusion:** The Article 5 prohibitions have applied since 2 February 2025, with Article 99 penalties available since 2 August 2025 and the enforcement machinery fully operational from 2 August 2026. No specific enforcement action under Article 5 is recorded in the available sources.""",

"q29": """**Issue:** Are there any recent articles or news on penalties imposed on companies under the EU AI Act?

**Rule:** Article 99 sets three tiers of administrative fine, in each case the higher of a fixed sum or a percentage of total worldwide annual turnover for the preceding financial year: up to EUR 35 000 000 or 7 % for breach of the Article 5 prohibitions; up to EUR 15 000 000 or 3 % for most other provider and deployer breaches, including the Article 50 transparency duties; and up to EUR 7 500 000 or 1 % for supplying incorrect, incomplete or misleading information to notified bodies or national competent authorities. Article 99(6) reverses the rule for SMEs and start-ups, for which each fine is capped at the lower of the two figures. Article 101 governs fines on providers of general-purpose AI models.

**Application:** On the available reporting, the Article 99 tiers have been enforceable since 2 August 2025, and commentary published in July 2026 confirms the three tiers and the SME reversal, noting that the figures are verified against Regulation (EU) 2024/1689 as amended by the Digital Omnibus. The enforcement machinery became operational on 2 August 2026, and from that date the AI Office may impose administrative fines on GPAI providers. The available sources set out how the penalty regime works and confirm that it is live, but they do not report any fine actually imposed on a named company.

**Conclusion:** The penalty ceilings are EUR 35 000 000 or 7 %, EUR 15 000 000 or 3 %, and EUR 7 500 000 or 1 %, with the lower figure applying to SMEs and start-ups, enforceable since 2 August 2025 and backed by operational enforcement powers from 2 August 2026. No penalty imposed on an identified company is recorded in the available sources.""",

"q30": """**Issue:** What is the latest news about standardisation requests for AI systems under the EU AI Act?

**Rule:** Article 40 provides that high-risk AI systems or general-purpose AI models conforming to harmonised standards whose references have been published in the Official Journal in accordance with Regulation (EU) No 1025/2012 are presumed to conform to the requirements of Chapter III Section 2 or the Chapter V obligations, to the extent those standards cover them. Article 40 further provides that when issuing a standardisation request to the European standardisation organisations, the Commission shall specify that standards be clear and consistent, including with standards developed for products covered by the Annex I harmonisation legislation, and shall request evidence of best efforts to fulfil the stated objectives. Article 41 allows the Commission to adopt common specifications by implementing act where harmonised standards are insufficient, where the standardisation request is not accepted, or where there are undue delays.

**Application:** On the available reporting, harmonised standards are presented as the mechanism that will supply legal certainty under the Act, support innovation and position the Union to set global benchmarks, and their delivery is treated as a Commission priority because the Chapter III requirements have to be made operational through them. The reported development is that CEN and CENELEC have decided to accelerate the development of the standards, which indicates that delivery has been behind the pace the timetable assumed. The practical significance is direct: under Article 43(1) the availability of harmonised standards determines whether a provider of an Annex III point 1 biometrics system may use the internal control route in Annex VI rather than the notified body route in Annex VII.

**Conclusion:** Standardisation rests on Article 40, with Article 41 common specifications as the fallback. The reported development is a CEN and CENELEC decision to accelerate delivery of the harmonised standards, against a timetable in which those standards gate the presumption of conformity and the choice of conformity assessment route.""",
}

# Statute-correct, but the corpus now reports the Omnibus deferral; without this
# a system citing 2 December 2027 would be marked down for being current.
Q12_NOTE = (
    "\n\n**Note on subsequent amendment:** The dates above are those in Article 113 "
    "of Regulation (EU) 2024/1689 as originally published. Reporting in the corpus "
    "records a Digital Omnibus political agreement of 7 May 2026 deferring the "
    "Annex III high-risk obligations from 2 August 2026 to 2 December 2027, with "
    "the Article 50 transparency obligations and full enforcement powers taking "
    "effect on 2 August 2026. An answer reflecting either the original Article 113 "
    "schedule or the reported deferral should be treated as correct."
)


def main() -> None:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    n_news = 0
    for item in data:
        qid = item["id"]
        if qid in NEWS:
            item["gold"] = NEWS[qid]
            item["gold_status"] = "revised_against_news_snapshot"
            item["needs_review"] = False
            item["gold_snapshot"] = SNAPSHOT
            n_news += 1
        elif qid == "q12" and "Note on subsequent amendment" not in item["gold"]:
            item["gold"] = item["gold"] + Q12_NOTE
            item["gold_snapshot"] = SNAPSHOT

    DATASET.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    flagged = [i["id"] for i in data if i["needs_review"]]
    print(f"rewrote {n_news} routing golds against the {SNAPSHOT}")
    print(f"q12 annotated with the reported Omnibus deferral")
    print(f"still flagged needs_review: {flagged or 'none'}")


if __name__ == "__main__":
    main()
