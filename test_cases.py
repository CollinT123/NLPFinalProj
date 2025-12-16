"""
Test cases for News Analysis API
Includes both misleading and non-misleading headline/article pairs
"""

test_cases = [
    # ========== MISLEADING HEADLINES ==========
    
    {
        "name": "Misleading - Clickbait Headline",
        "category": "misleading",
        "headline": "BREAKING: Stock Market Crashes 50% Overnight!",
        "article": "The stock market experienced a minor correction today, with the S&P 500 dropping 0.5% in after-hours trading. Analysts suggest this is a normal market fluctuation and expect recovery by next week. The Federal Reserve has not made any emergency announcements."
    },
    
    {
        "name": "Misleading - Exaggerated Claims",
        "category": "misleading",
        "headline": "Scientists Discover Cure for All Diseases",
        "article": "Researchers at a local university have made progress in understanding cellular regeneration. The preliminary study, conducted on mice, shows promising results for treating certain types of inflammation. Clinical trials on humans are still years away, and the research is in early stages."
    },
    
    {
        "name": "Misleading - Out of Context",
        "category": "misleading",
        "headline": "President Announces Immediate Tax Increases",
        "article": "During a press conference, the President discussed long-term economic policy options. When asked about potential tax reforms, he mentioned that various proposals are being considered for the next fiscal year. No immediate changes were announced, and any proposals would require congressional approval."
    },
    
    {
        "name": "Misleading - Sensationalism",
        "category": "misleading",
        "headline": "Local Restaurant Shuts Down After Health Violations",
        "article": "A popular local restaurant temporarily closed for scheduled renovations this week. The owners announced they are updating their kitchen equipment and expanding seating capacity. The restaurant is expected to reopen next month with improved facilities."
    },
    
    {
        "name": "Misleading - Misleading Statistics",
        "category": "misleading",
        "headline": "Unemployment Rate Doubles in One Month",
        "article": "The unemployment rate increased from 3.5% to 3.6% this month, according to the latest labor statistics. Economists note this is within normal seasonal variation. The overall job market remains strong, with continued job growth in technology and healthcare sectors."
    },
    
    {
        "name": "Misleading - False Attribution",
        "category": "misleading",
        "headline": "Tech CEO Predicts Economic Collapse",
        "article": "At a technology conference, a panel discussion covered various economic topics. One participant, a startup founder, shared personal opinions about market trends during an informal Q&A session. The comments were not part of any official statement or company position."
    },

   #TEST CASE
    {
        "name": "Misleading - Headline Doesn't Match Content",
        "category": "misleading",
        "headline": "New Study Proves Coffee Causes Cancer",
        "article": "A recent study examined the health effects of coffee consumption. The research found no significant correlation between moderate coffee intake and cancer risk. In fact, the study suggested potential health benefits from antioxidants found in coffee. Researchers recommend 2-3 cups per day as part of a balanced diet."
    },
    
    {
        "name": "Misleading - Missing Context",
        "category": "misleading",
        "headline": "Major City Declares Bankruptcy",
        "article": "A small municipality with a population of 5,000 residents filed for financial restructuring assistance. This is a routine procedure for small towns facing budget challenges. The city is working with state officials to develop a recovery plan. This does not affect any major metropolitan areas."
    },
    
    # ========== NON-MISLEADING HEADLINES ==========
    
    {
        "name": "Accurate - Straightforward News",
        "category": "non-misleading",
        "headline": "Federal Reserve Raises Interest Rates by 0.25%",
        "article": "The Federal Reserve announced today that it will raise the federal funds rate by 0.25 percentage points, bringing the target range to 4.5% to 4.75%. This marks the tenth rate increase since March 2022. Fed Chair Jerome Powell stated that the decision reflects ongoing efforts to combat inflation while monitoring economic growth."
    },
    
    {
        "name": "Accurate - Research Findings",
        "category": "non-misleading",
        "headline": "Study Links Regular Exercise to Improved Mental Health",
        "article": "A comprehensive study published in the Journal of Health Psychology found that individuals who engage in at least 150 minutes of moderate exercise per week report significantly better mental health outcomes. The study surveyed over 10,000 participants across different age groups. Researchers observed reduced anxiety and depression symptoms in the active group compared to sedentary participants."
    },

    # Use for Testing
    {
        "name": "Accurate - Economic Report",
        "category": "non-misleading",
        "headline": "GDP Growth Slows to 2.1% in Q3",
        "article": "The U.S. economy grew at an annualized rate of 2.1% in the third quarter, according to data released by the Commerce Department. This represents a slowdown from the 2.6% growth rate in the previous quarter. Economists attribute the deceleration to reduced consumer spending and weaker export performance. The growth rate remains positive, indicating continued economic expansion."
    },
    
    {
        "name": "Accurate - Policy Announcement",
        "category": "non-misleading",
        "headline": "City Council Approves New Affordable Housing Initiative",
        "article": "The City Council voted unanimously yesterday to approve a $50 million affordable housing initiative. The program will provide funding for the construction of 500 new housing units over the next three years. Mayor Johnson stated that the initiative addresses the city's growing housing crisis and will help low-income families find stable housing. Construction is expected to begin next spring."
    },
    
    {
        "name": "Accurate - Technology News",
        "category": "non-misleading",
        "headline": "Tech Company Announces New AI Product Launch",
        "article": "TechCorp announced today the launch of its new artificial intelligence assistant, designed to help businesses automate customer service tasks. The product will be available starting next month with pricing starting at $99 per month. The company's CEO highlighted the product's ability to handle complex queries and integrate with existing business software. Early beta testers reported positive feedback on the system's accuracy and response time."
    },
    
    {
        "name": "Accurate - Health Update",
        "category": "non-misleading",
        "headline": "New Vaccine Shows 85% Effectiveness in Clinical Trials",
        "article": "Pharmaceutical company MedPharm reported that its new vaccine candidate demonstrated 85% effectiveness in preventing infection during Phase 3 clinical trials. The study involved 15,000 participants across multiple countries. Side effects were reported as mild, primarily including temporary fatigue and injection site soreness. The company plans to submit the vaccine for regulatory approval next month."
    },
    
    {
        "name": "Accurate - Weather Report",
        "category": "non-misleading",
        "headline": "Hurricane Warning Issued for Coastal Regions",
        "article": "The National Weather Service has issued a hurricane warning for coastal areas from Florida to North Carolina. Hurricane Maria, currently a Category 3 storm, is expected to make landfall within 48 hours. Residents are advised to evacuate low-lying areas and prepare emergency supplies. Wind speeds are projected to reach 120 miles per hour at landfall."
    },
    
    {
        "name": "Accurate - Sports News",
        "category": "non-misleading",
        "headline": "Local Team Wins Championship After Overtime Victory",
        "article": "The city's basketball team secured the state championship last night with a thrilling 98-95 overtime victory. The game went into overtime after both teams were tied at 89 points at the end of regulation. Star player John Smith scored 35 points, including the game-winning three-pointer with 2 seconds remaining. This marks the team's first championship in 15 years."
    },
    
    {
        "name": "Accurate - Business News",
        "category": "non-misleading",
        "headline": "Company Reports 12% Revenue Increase in Q4",
        "article": "GlobalCorp announced its fourth-quarter earnings today, reporting a 12% increase in revenue compared to the same period last year. The company's revenue reached $2.5 billion, driven primarily by strong performance in its technology division. CEO Jane Doe attributed the growth to successful product launches and expanded market presence. The company also announced plans to increase its workforce by 10% in the coming year."
    },
    
    {
        "name": "Accurate - Education News",
        "category": "non-misleading",
        "headline": "University Receives $10 Million Grant for Research",
        "article": "State University announced it has received a $10 million grant from the National Science Foundation to fund research in renewable energy technologies. The five-year grant will support the work of 20 researchers and graduate students. The research will focus on improving solar panel efficiency and developing new battery storage solutions. University President Dr. Robert Lee expressed gratitude for the funding, which will advance the institution's commitment to sustainability research."
    },
    
    # ========== EDGE CASES ==========
    
    {
        "name": "Edge Case - Very Short Headline",
        "category": "edge_case",
        "headline": "Market Up",
        "article": "The stock market closed higher today, with the Dow Jones Industrial Average gaining 150 points. Technology stocks led the gains, with major tech companies seeing increases of 2-3%. Analysts attribute the positive movement to strong quarterly earnings reports and optimistic economic forecasts."
    },



]


