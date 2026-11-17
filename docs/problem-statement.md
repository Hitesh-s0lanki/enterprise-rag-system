# Problem Statement: Enterprise IT Support Knowledge Copilot

## Background

NovaRetail, a fictional organization with 3,000 employees, maintains an internal IT knowledge base covering VPN access, password rules, multi-factor authentication (MFA), software installation, laptop troubleshooting, and service-desk procedures.

Despite having this information, employees continue to raise repetitive support tickets because they struggle to find the right document or identify the current, applicable guidance. Keyword search often returns many documents without a clear answer. A general chatbot can provide confident responses that are unsupported by company policy, while static internal documentation may lack current information about external service incidents.

## Core Problem

Employees need fast, clear answers to IT support questions, grounded in trustworthy and current information they are authorized to access. The organization needs to verify answer quality, identify supporting sources and their freshness, measure retrieval performance, and keep existing knowledge accurate through controlled updates. Development and ongoing usage must remain cost-efficient, with controlled token consumption and visibility into spending.

The challenge is to make company knowledge useful and accessible while preventing unsupported answers, unauthorized disclosure, and unauthorized changes to the knowledge base.

## Users

- **Employees:** Need reliable answers to routine IT questions with clear supporting sources.
- **IT support staff:** Need to reduce repetitive tickets and identify missing, outdated, or conflicting guidance.
- **Authorized knowledge contributors and administrators:** Need to maintain existing knowledge within the permissions assigned to their roles.

## Required Capabilities

### 1. Reliable Answers from Company Knowledge

- Prioritize the private company knowledge base when answering employee questions.
- Evaluate whether retrieved information is relevant, sufficient, and applicable before using it to answer.
- When evidence is weak, support refining the question and retrying retrieval.
- Where internal information is missing or outdated, allow controlled use of credible public sources for questions that require current external information.
- Clearly distinguish external guidance from internal company policy. External information must not override company policy or expose private company information.
- If reliable evidence remains unavailable, state the limitation and direct the employee to IT support rather than provide an unsupported answer.
- Respect the employee's access permissions when retrieving information and presenting answers or sources.

### 2. Retrieval Latency, Output Grading, Sources, and Update Timestamps

- **Document retrieval latency:** Record and expose the time taken to retrieve supporting documents for each query, separately from total response time, so retrieval delays can be identified.
- **Output grading:** Evaluate the final answer for relevance, completeness, and factual support in the cited evidence. Make the grading result and a brief reason available for review, and flag answers that do not meet the agreed quality criteria.
- **Sources:** Provide traceable references to the documents or public pages supporting the answer, including the relevant section or page where available. Identify whether each source is internal or external.
- **Update timestamps:** Show the last-updated timestamp for each supporting source where available, distinguishing it from retrieval time. If a source's update timestamp is unavailable, explicitly label it as unknown.
- Make the evidence assessment and a concise record of retrieval, fallback, and retry decisions available for troubleshooting and quality review.

### 3. Role-Based Updates to the Existing Knowledge Base

- Support adding documents and updating or replacing existing knowledge as policies and procedures change.
- Enforce role-based access control (RBAC) for knowledge maintenance. A user's role and assigned scope must determine which documents they can add, edit, replace, or approve where approval is required.
- Separate permission to read knowledge from permission to modify it. Deny unauthorized update attempts.
- Ensure accepted updates become available to subsequent queries and superseded content is no longer presented as current guidance.
- Preserve revision history and record who changed a document, what changed, and when the update occurred.
- Maintain source references, access permissions, and last-updated timestamps across revisions so answers remain traceable to the version used.

### 4. Cost and Token Efficiency

- **During development:** Track and optimize token usage and costs incurred while building, testing, evaluating, and refining the solution, including initial knowledge ingestion. Avoid unnecessary repeated processing and model calls.
- **During employee queries:** Minimize unnecessary input and output tokens, repeated retrieval, model calls, and external searches while preserving answer quality, required grading, access controls, and acceptable response times.
- **During knowledge updates:** Avoid reprocessing unchanged knowledge unnecessarily, and account for the cost of keeping existing content current.
- **Usage visibility:** Record input and output token usage and attributable cost per query, including evidence evaluation, answer generation, output grading, and any retries. Distinguish measured charges from estimates and include retrieval, external search, and other service costs where applicable.
- **Cost oversight:** Report development spending separately from ongoing operating costs, with aggregate usage and cost per successfully answered query available to authorized reviewers.
- **Budget controls:** Support agreed spending and token limits for development and runtime use, including bounded retries. When a limit prevents a reliable answer, explain the limitation rather than silently reducing required quality checks or producing an unsupported response.

## Representative Scenarios

- **VPN access:** An employee asks how to connect from home and receives instructions supported by the current internal VPN guide, with its source, update timestamp, retrieval latency, and answer grade available.
- **Current service incident:** An employee asks about the latest Microsoft Teams outage. If internal guidance is insufficient, credible external information may support an answer, with the source and freshness clearly identified.
- **Knowledge update:** An authorized IT contributor revises an existing VPN guide within their assigned scope. Subsequent answers use the accepted revision, while users without update permission cannot change the guide. The change remains auditable.
- **Cost-efficient operation:** Development evaluations and employee queries have separately tracked token usage and costs. Routine questions use only the necessary processing, while more complex queries remain within agreed budgets without bypassing evidence or output grading.

## Expected Outcomes

- Fewer repetitive IT support tickets and less time spent searching for guidance.
- Answers supported by relevant evidence, with visible quality assessments and source freshness.
- Measurable retrieval performance and identifiable latency bottlenecks.
- Current knowledge maintained by authorized users, with traceable revisions and enforced access boundaries.
- Controlled development and operating costs, with measurable token efficiency and visibility into the cost of answering employee queries.

Retrieval latency targets, output grading thresholds, cost budgets, and token limits must be agreed with the business before acceptance; this problem statement does not prescribe numerical targets.
