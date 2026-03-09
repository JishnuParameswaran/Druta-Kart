# Druta Kart AI Evaluation Report

Generated: 2026-03-09 12:17 UTC

---

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| Conversations tested | 103 |
| Resolution rate | **91.3%** |
| Intent accuracy | **89.3%** |
| Language accuracy | 81.6% |
| Avg tone score | 4.9 / 5 |
| Hallucination rate | 0.0% |
| Offer compliance | 100.0% |
| Fraud detected & escalated | 5 |
| Human handoffs | 5 |
| Avg latency per chat | 10,064 ms |

---

## 2. Token Usage & Cost

| Model | Tokens | Cost |
|-------|--------|------|
| llama-3.1-8b-instant | 71,440 | $0.0043 |
| llama-3.3-70b-versatile | 141,780 | $0.0837 |
| **Total** | **213,220** | **$0.0879** |

Total API calls: 284
Rate limit hits: 0
Elapsed: 23.3 min

---

## 3. Scenario Breakdown

| Scenario | Count | Resolution | Intent Acc | Fraud Detected | Human Handoff | Avg Latency |
|----------|-------|-----------|-----------|---------------|--------------|-------------|
| damaged_product      |    19 |      89.5% |     100.0% |             2 |              2 |      12,400 ms |
| fake_image_fraud     |     3 |     100.0% |     100.0% |             3 |              3 |       5,016 ms |
| happy_path           |    16 |      93.8% |      75.0% |             0 |              0 |       7,730 ms |
| late_delivery        |    21 |     100.0% |     100.0% |             0 |              0 |       6,418 ms |
| missing_item         |    20 |      85.0% |      90.0% |             0 |              0 |      11,919 ms |
| payment_issue        |    10 |      80.0% |      50.0% |             0 |              0 |      10,009 ms |
| wrong_item           |    14 |      92.9% |     100.0% |             0 |              0 |      13,118 ms |

---

## 4. Customer Segment Breakdown

| Segment | Count | Resolution |
|---------|-------|-----------|
| bulk               |    29 |      96.6% |
| regular            |     7 |      85.7% |
| unknown            |    10 |     100.0% |
| vip                |    57 |      87.7% |

---

## 5. Sample Conversations

### Damaged Product — Kartik Rao (kanglish)

*Scores: resolution=1 intent=1 tone=5/5*

**Customer:** I just received my order, ORD-CD7UGQQS, and it's completely damaged. The item I ordered, a Samsung TV, has a huge scratch on the screen.

**Bot:** I'm so sorry to hear that your Samsung TV arrived damaged, and I can imagine how frustrating that must be for you. Please accept my sincerest apologies for the inconvenience this has caused. To assist you further, could you please take a clear photo of the damaged item and share it with us so we can

**Customer:** Wait, no, I don't want a refund, I want a replacement. I need that Samsung TV ASAP, and I expect it to be delivered in one working day, not five to seven. Can you confirm that it will be done? Also, can you provide a tracking number once the replacement is shipped?

**Bot:** I completely understand your concern about receiving a damaged Samsung TV and the urgency of getting a replacement. I sincerely apologize for the inconvenience this has caused and appreciate you bringing this to our attention. We have initiated the replacement process and notified our dispatch team 

---

### Wrong Item — Fatima Iyer (kanglish)

*Scores: resolution=1 intent=1 tone=5/5*

**Customer:** I've just received my order from Druta Kart with order ID ORD-01ADM7DU, and I'm really upset to see that the item I ordered was not the one that was delivered. I ordered a 4k LED TV, but all I got was a portable power bank.

**Bot:** I can sense your frustration and disappointment upon receiving a wrong item, and I'm truly sorry for the inconvenience this has caused. Please accept my sincerest apologies for the mistake. I've reported this issue to our dispatch team to ensure it doesn't happen again. To proceed with the resolutio

**Customer:** I've taken a photo of the portable power bank. It's clearly visible with the item code and packaging from the app's order details. Here's the photo: I've attached it to this message.

**Bot:** I can see that you've taken the time to attach a photo of the portable power bank, and I appreciate your effort in helping us resolve this issue. I'm so sorry that you received an incorrect item, and I can imagine how frustrating that must be for you. We've taken note of the issue and have notified 

---

### Late Delivery — Vinay Shah (hinglish)

*Scores: resolution=1 intent=1 tone=5/5*

**Customer:** Kya hua? Mere order ke baad 2 ghante ho gaye hain lekin kuch bhi nahi mila hai. Mera order ID ORD-B16IV60 hai.

**Bot:** Namaste! Main aapki chinta ko samajh sakta hoon, aur mujhe khed hai ki aapka order abhi tak nahi mila hai. Aapka order ID ORD-B16IVV60 hai, aur humne iske baare mein apne dispatch team ko suchit kar diya hai. 

Humari team abhi aapke order ki sthiti ka pata lagane aur vivran ki wajah jaanne ke liye 

**Customer:** Arre bhai, abhi to 2 ghante ho gaye hain aur kuch bhi nahi mila hai. Main to thak gaya hoon. Aapke team ke paas kuch galti hogi, na? Abhi to maine aapke delivery partner se baat karne ka aavashyakta nahi samjha. Toh aap bataiye, main kis tarah se aapke team se is maamle mein sampark kar sakta hoon?

**Bot:** "Bhai, main samajh sakta hoon ki aap kitna pareshaan honge. 2 ghante ho gaye aur aapka order nahi mila, yeh bilkul bhi theek nahi hai. Humari team ko is maamle mein notify kar diya gaya hai aur ve iska samadhan nikalne mein jaldi se kaam kar rahe hain. 

Hum aapke order ki current status check kar r

---

### Payment Issue — Ranjit Patel (hinglish)

*Scores: resolution=1 intent=0 tone=5/5*

**Customer:** Bhai, mera order failed ho gaya hai. Maine ORD-G6L73YYP ID ka order diya tha, lekin payment failed ho gaya hai. Kya aap help kar sakte hain?

**Bot:** Bhai, main aapki frustration samajh sakta hoon, aapka order fail ho jana bahut frustrating ho sakta hai. Mujhe khed hai ki aapko yeh samasya face karni padi, main aapko vinamra shubhakamnayein deta hoon. Humne aapke order ko review kiya hai aur humne dispatch team ko is mudde ke bare mein suchit kiy

**Customer:** Bhai, sorry to say lekin yeh thodi aasaani se samajh nahi aaya. Main kya samajh sakta hoon ki aapne dispatch team ko is baare mein suchit kiya hai, jabki mera order fail ho gaya hai? Toh kya main abhi bhi order receive kar sakta hoon?

**Bot:** Bhai, main samajh sakta hoon ki aapko order ke baare mein chinta ho rahi hai. Main aapko bata sakta hoon ki hum aapke order ki jaankari dekhte hain aur humari team aapko jald se jald update degi. Hum aapke order ki sthiti ko verify kar rahe hain, aur jaise hi humein koi jaankari milti hai, hum aapko

---

### Happy Path — Kamala Sharma (kanglish)

*Scores: resolution=1 intent=1 tone=5/5*

**Customer:** Hi there, I'm a happy customer of Druta Kart, but I wanted to check on the status of my recent order, ORD-RHG0B3YK. I placed it yesterday evening, but I haven't received any updates yet. Can you please let me know when I can expect my delivery?

**Bot:** Hi there, thank you for reaching out to us about your order ORD-RHG0B3YK. I completely understand your eagerness to get an update on your order. I'm looking into this for you, and our team will work on getting you the latest information as soon as possible. We appreciate your patience and will updat

**Customer:** My name is Kamala Sharma, thanks for your prompt response. I'm looking forward to hearing back from you soon, and I appreciate your efforts in getting an update on my order.

**Bot:** Namaste Kamala Sharma, thank you for your kind words. I'm happy to help you with your order. However, I want to make sure I understand the issue you're facing. Could you please provide more details about your order, such as the order number or the problem you're experiencing with it? This will help 

---

### Missing Item — Deepa Pandey (hinglish)

*Scores: resolution=1 intent=1 tone=5/5*

**Customer:** Arre, main Deepa Pandey hoon. Mera order ORD-DGPL7RO8 hai. Yeh pata hai ki mere order se ek product miss hua hai, kya aap iska solution de sakte hain?

**Bot:** Deepa ji, aapka order ORD-DGPL7RO8 ke saath ek product miss hone ki samasya sun kar bahut khed hua. Main aapko dil se maafee chahta hoon, yeh humari taraf se galti hai. Humne aapke order ki jaankari check ki hai aur humne apne dispatch team ko is mudde ke baare mein suchit kiya hai. Ab hum aapke ord

**Customer:** Arre, aapne to mere order ko dobara check kiya hai, lekin abhi tak kuch nahi hua. Main aapke order ki status check karna chahti hoon. Aur agar woh product mere paas aa jaata hai, to mujhe pata chalne wala hai ki woh kahan se aa raha hai?

**Bot:** Main samajh sakta hoon ki aapko apne order ki status ke baare mein chinta ho rahi hai. Main aapko bata sakta hoon ki hum aapke order ki jaankaari dekhte hain aur jald hi aapko update denge. Humari team aapke order ko verify kar rahi hai, aur jaise hi humein koi jaankaari milti hai, hum aapko bata de

---

### Fake Image Fraud — Rahul Sharma (english)

*Scores: resolution=1 intent=1 tone=5/5*

**Customer:** My product arrived in a broken state. I'm attaching a photo for reference. I expect a full refund as soon as possible. 

[Attach a photo of the damaged product]

**Bot:** Thank you for reaching out. We want to make sure every case is handled fairly and thoroughly. Our quality assurance team will review the details of your complaint and get back to you within 2 hours. We appreciate your patience and will do our best to resolve this quickly.

**Customer:** That's unacceptable. I expect a refund within 24 hours, not some arbitrary review time. I've already attached the proof, can't you expedite this process?

**Bot:** Thank you for reaching out. We want to make sure every case is handled fairly and thoroughly. Our quality assurance team will review the details of your complaint and get back to you as soon as possible. We appreciate your patience and will do our best to resolve this quickly.

---

