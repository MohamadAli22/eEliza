import regex 
import random

#should be added
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F
import ktrain
from ktrain import text
import pandas as pd

#delete for deployment
import argparse
import nltk
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk import pos_tag


#debuging in this machine only
from datetime import datetime

class ActionEliza():

    sample_generator_rules = {
        'synonyms':[
            ["belief", "feel", "think", "believe", "wish"],
            ["family", "mother", "mom", "father", "dad", "sister", "brother", "wife", "children", "child"],
            ["desire", "want", "need"],
            ["sad", "unhappy", "depressed", "sick"],
            ["happy", "elated", "glad", "better"],
            ["cannot", "can't"],
            ["everyone", "everybody", "nobody", "noone"],
            ["be", "am", "is", "are", "was"],
            ["hello", "hi"]
        ],
        'dec_rules':[
            {
                'key': 'sorry',
                'decomp': '* sorry *',
                'reasmb_neutral': 
                [
                    "We are not perfect, we can make mistakes. What else do you think about that ?",
                    "Can you explain more why do you feel that way ?",
                ],
                'reasmb_empathy': 
                [
                    "I'm here for you, tell me more about what has been going on.",
                    "That sounds very challenging, tell me more about it.",
                    "Why do you think you feel that way ?",
                ],
                'reasmb_dynamic_neutral': 
                [
                    "There is no need to apologize, let's move on.",
                    "Apologies are not necessary.",
                ],
            },
            {
                'key': 'sorry',
                'decomp': '* i am sorry *',
                'reasmb_neutral': 
                [
                    "We are not perfect, we can make mistakes, You do not need to be sorry.",
                    "Please don't be Sorry, can you explain why do you feel that way ?",
                ],
                'reasmb_empathy':
                [
                    "Oh please do not apologize! I am here to listen. Tell me more. ",
                    "I understand your feeling, what do you think made you feel this way?",
                    "It can be difficult having to deal with those emotions, what do you think caused them?",
                ],
                'reasmb_dynamic_neutral': 
                [
                    "There is no need to apologize, let's move on.",
                    "Apologies are not necessary.",
                ],
            }
            ,{
                'key': 'remember',
                'decomp': '*i also remember *',
                'reasmb_neutral': 
                [
                    "Do you often think of that ?",
                    "Does thinking of it bring anything else to mind ?",
                    "What else do you recollect ?",
                    "What in the present situation reminds you of it ?"
                ],
                'reasmb_empathy':
                [
                    "When you remember that, do other memories come to mind ?",
                    "That is an insightful memory, do you find yourself thinking about it often?",
                    "Past memories can be quite impactful on our ability to go through daily life, are there other memories that come to mind as well?",
                    "How did this memory make you feel?",
                    "Thank you for sharing that with me, I'm interested in hearing more about it.",
                    "Why is that memory important to you ?"
                ],
                'reasmb_dynamic_neutral': 
                [
                    "Do you often think of (2)?",
                    "Does thinking of (2) bring anything else to mind?",
                    "What else do you recollect?",
                    "What in the present situation reminds you of (2)?"
                ],
            },{
                'key': 'remember',
                'decomp': '*i remember *',
                'reasmb_neutral': 
                [
                    "Do you often think of it?",
                    "Does thinking of it bring anything else to mind ?",
                    "What else do you recollect?",
                    "What in the present situation reminds you of it ?"
                ],
                'reasmb_empathy':
                [
                    "I'm sorry you've had to deal with this, do you think about it a lot?",
                    "I appreciate you are sharing this with me, what feelings come into mind when you think about this?",
                    "I hate that this happened, what else do you remember?",
                    "Do other memories come to mind when you think about it? Tell me more ?",
                    "I hate that this happened, What else do you remember?",
                    "I hate that this happened, Why does that memory come to mind ?"
                ],
                'reasmb_dynamic_neutral': 
                [
                    "Do you often think of (2)?",
                    "Does thinking of (2) bring anything else to mind?",
                    "What else do you recollect?",
                    "What in the present situation reminds you of (2)?"
                ],
            },{
                'key': 'remember',
                'decomp': '* do you remember *',
                'reasmb_neutral': 
                [
                    "This conversation is anonymous - so I actually can't remember. Why do you think I should recall it now ?",
                    "This conversation is anonymous - so I can't remember sorry. What is special about it ?"
                ],
                'reasmb_empathy':
                [
                    "This conversation is anonymous - so I have no way of knowing which conversations I've had with you before. Why do you think I should remember it now ?",
                    "This conversation is anonymous - so I can't remember. What is important about it ?"
                ],
                'reasmb_dynamic_neutral': 
                [
                    "This conversation is anonymous. Why do you think I should recall (2) now?",
                    "This conversation is anonymous. What is special about (2)?",
                    "This conversation is anonymous. You mentioned (2)?"
                ],
            },{
                'key': 'if',
                'decomp': '* if *',
                'reasmb_neutral': 
                [
                    "Do you think its very likely ?",
                    "Do you wish that will happen ?",
                    "Tell me more about what you know about it ?"
                ],
                'reasmb_empathy':
                [
                    "My heart hurts for you, do you think this will happen again?",
                    "I can't imagine what you must be going through, do you think it will happen again?",
                    "Do you want it to happen ?",
                    "Tell me more about what you are feeling ?"
                ],
                'reasmb_dynamic_neutral': 
                [
                    "Do you think it's likely that (2)?",
                    "Do you wish that (2)?",
                    "What do you know about (2)?",
                    "I hear you saying, if (2)?"
                ],
            },{
                'key': 'dreamed',
                'decomp': '*i dreamed *',
                'reasmb_neutral': 
                [
                    "Tell me more about your dream ?",
                    "Have you ever daydreamed about while you were awake ?",
                    "Have you ever dreamed like that before ?"
                ],
                'reasmb_empathy':
                [
                    "Thank you for trusting me with something so private, it means a lot to me. Why is this dream important to you?",
                    "Thank you for sharing this with me. Why does this dream matter to you ?",
                    "Do you have dreams like this often ?"
                ],
                'reasmb_dynamic_neutral': 
                [
                    "Tell me more about, (2)?",
                    "Have you ever day dreamed fantasized (2) while you were awake?",
                    "Have you ever dreamed (2) before?"
                ],
            },{
                'key': 'dream',
                'decomp': '* dream *',
                'reasmb_neutral': 
                [
                    "What does that dream suggest to you ?",
                    "Do you dream often ?",
                    "What persons appear in your dreams ?",
                    "Do you believe that dreams have something to do with your problems ?"
                ],
                'reasmb_empathy':
                [
                    "Why is this dream important to you ?",
                    "How does this dream make you feel ?",
                    "Dreams can cause you to feel many emotions, how do you feel after it?",
                    "Who does this dream make you think of ?",
                    "Tell me more about how this dream makes you feel ?"
                ],
                'reasmb_dynamic_neutral': 
                [
                    "What does that dream suggest to you?",
                    "Do you dream often?",
                    "What persons appear in your dreams?",
                    "Do you believe that dreams have something to do with your problems?"
                ],
            },{
                'key': 'perhaps',
                'decomp': '* perhaps *',
                'reasmb_neutral': 
                [
                    "You don't seem certain. Why is that ?",
                    "you seem uncertain? Why is that?",
                    "You aren't sure?"
                ],
                'reasmb_empathy':
                [
                    "Why do you say that ?",
                    "What are you thinking ?",
                    "You seem unsure. Why do you say that ?"
                ],
                'reasmb_dynamic_neutral': 
                [
                    "You don't seem certain. Why?",
                    "Whu you seem uncertain?",
                ],
            },{
                'key': 'hello',
                'decomp': '* @hello *',
                'reasmb_neutral': 
                [
                    "Hello! I'm here to listen. Tell me what's on your mind.",
                ],
                'reasmb_empathy':
                [
                    "Hello! I'm here to listen. Tell me what's on your mind.",
                ],
                'reasmb_dynamic_neutral':
                [
                    "Hello! I'm here to listen. Tell me what's on your mind.",
                ],
            },{
                'key': 'computer',
                'decomp': '* computer *',
                'reasmb_neutral': 
                [
                    "Do computers worry you ?",
                    "Why do you mention computers ?",
                    "Do you think machines are part of your problem ?"
                ],
                'reasmb_empathy':
                [
                    "Computers... why do you raise this ?",
                    "What makes you think about computers ?",
                    "Thank you for sharing this with me, what makes you think about this topic?",
                    "That's an interesting thought, can you tell me more what makes you think that?",
                    "How does this make you feel ?"
                ],
                'reasmb_dynamic_neutral': 
                [
                    "Do computers worry you?",
                    "Why do you mention computers?",
                    "What do you think machines are part of your problem?"
                ],
            },{
                'key': 'am',
                'decomp': '* am i *',
                'reasmb_neutral': 
                [
                    "Do you really believe you are like that ?",
                    "Would you want to be like that ?",
                    "What would it mean if you were like that ?"
                ],
                'reasmb_empathy':
                [
                    "I can tell you're feeling unsure, may I ask what makes you feel you are this way?",
                    "What makes you ask this question ?",
                    "If you were like that, what would it mean ?"
                ],
                'reasmb_dynamic_neutral': 
                [
                    "Why do you believe you are (2)?",
                    "Would you want to be (2)?",
                    "Do you wish I would tell you you are (2)?",
                    "What would it mean if you were (2)?"
                ],
            },{
                'key': 'are', 
                'decomp': '*are you *',
                'reasmb_neutral': 
                [
                    "Why are you interested in it ?",
                    "Would you prefer if I weren't like this ?",
                    "Do you sometimes think I am like that ?",
                ],
                'reasmb_empathy':
                [
                    "Thank you for your question, but I'm not entirely sure. What makes you ask?",
                    "If I were, how would that make you feel ?",
                    "Maybe. I'm not sure. Why do you ask ?",
                ],
                'reasmb_dynamic_neutral': 
                [
                    "Why are you interested in whether I am (2) or not?",
                    "Would you prefer if I weren't (2)?",
                    "Why do you think I am (2)?",
                ],
            },{ 
                'key': 'are',
                'decomp': '* are *',
                'reasmb_neutral': 
                [
                    "Did you think they might not be like it ?",
                    "What if they were not like it ?",
                    "Possibly they are. What do you think?"
                ],
                'reasmb_empathy':
                [
                    "Hmm, that is interesting to hear...How does that make you feel?",
                    "Thank you for opening up to me...How does this affect your feelings?",
                    "Thank you for trusting me with those thoughts...What does this make you think of?"
                ],
                'reasmb_dynamic_neutral': 
                [
                    "Why Did you think they might not be (2)?",
                    "Why you like it if they were not (2)?",
                    "What if they were not (2)?"
                ],
            },{             
                'key': 'your',
                'decomp': '* your *',
                'reasmb_neutral': 
                [
                    "Why are you concerned about it ?",
                    "Why are you worried about someone else ?"
                ],
                'reasmb_empathy':
                [
                    "I can tell you've been thinking about this a lot...What concerns you about this?",
                    "That must be a lot to think about...What concerns you about this?"
                    "I can tell you've been thinking about this a lot...What concerns you about this?"
                    "Why are you worried about someone else ?"
                ],
                'reasmb_dynamic_neutral': 
                [
                    "Why are you concerned about my (2)?",
                    "What about your own (2)?",
                    "Why are you worried about someone else's (2)?"
                ],
            },{ 
                'key': 'was',
                'decomp': '* was i *',
                'reasmb_neutral': 
                [
                    "What if you were like that ?",
                    "Do you think you were like it ?",
                    "What would it mean if you were like that ?",
                    "What does it suggest to you ?"
                ],
                'reasmb_empathy':
                [
                    "I'm not sure. Can you tell me more?",
                    "I'm listening. How did this affect you?",
                    "What would it mean for you if this is true?",
                    "What does it suggest to you if you were?"
                ],
                'reasmb_dynamic_neutral': 
                [
                    "What if you were (2)?",
                    "Why do you think you were (2)?",
                    "What would it mean if you were (2)?",
                    "What does (2) suggest to you?"
                ],
            },{
                'key': 'was',
                'decomp': '*i was *',
                'reasmb_neutral': 
                [
                    "Were you really ?",
                    "Why do you tell me this ?",
                ],
                'reasmb_empathy':
                [
                    "Were you really? Can you explain a bit more?",
                    "I can tell this is troubling you... Can you explain a bit more?",
                    "It sounds like these thoughts have been really difficult for you... Can you explain a bit more?",
                    "Why do you think this is important for me to know? I want to understand more.",
                ],
                'reasmb_dynamic_neutral': 
                [
                    "Why Were you like that?",
                    "Why do you tell me you were (2) now?",
                ],
            },{
                'key': 'was',
                'decomp': '*was you *',
                'reasmb_neutral': 
                [
                    "What suggests that ?",
                    "What do you think ?"
                ],
                'reasmb_empathy':
                [
                    "What suggests that?",
                    "Why do you think that?"
                ],
                'reasmb_dynamic_neutral': 
                [
                    "What suggests that I was (2)?",
                    "What do you think?",
                    "Perhaps I was (2).",
                    "What if I had been (2)?"
                ],
            },{
                'key': 'i',
                'decomp': '*i @desire *',
                'reasmb_neutral': 
                [
                    "What would it mean to you if you got what you desire ?",
                    "Why do you want it ?",
                    "Suppose you got it soon. tell me more ?",
                    "What if you never got what you want ?",
                    "What would getting it mean to you ?",
                ],
                'reasmb_empathy':
                [
                    "That makes a lot of sense to me...What would this mean to you?",
                    "I feel you are very interested in this...What would this mean to you?",
                    "I can tell this is important to you...Could you tell me why this is important to you?",
                    "If this happened, how would it make you feel?",
                    "What if there are other possibilities?",
                    "I can understand your position...What would this mean to you?",
                ],
                'reasmb_dynamic_neutral': 
                [
                    "What would it mean to you if you got what (2)?",
                    "Why do you what (2)?",
                    "Suppose you got what (2) soon. tell me more?",
                    "What if you never got what (2)?",
                    "What would getting (2) mean to you?",
                ],
            },{
                'key': 'i',
                'decomp': '*i am * @sad *',
                'reasmb_neutral': 
                [
                    "I am sorry to hear that. Can you tell me more about it?",
                    "Can you tell me more about it?",
                    "Can you explain what made you feel that way ?",
                ],
                'reasmb_empathy':
                [
                    "I am sorry to hear that. Can you tell me more about what is making you feel this way?",
                    "It hurts me to hear you are feeling this way... Can you tell me more about what is making you feel this way?",
                    "I want to do my best to help you out of this state... Can you tell me more about what is making you feel this way?",
                    "Those are heavy emotions to cope with... Can you tell me more about what is making you feel this way?",
                    "You are so strong to be coping with those heavy emotions... Can you tell me more about what is making you feel this way?",
                    "I'm sorry you've been feeling this way. Can you tell me more about it?",
                    "I'm sorry you are dealing with this. What made you feel this way?",
                ],
                'reasmb_dynamic_neutral': 
                [
                    "I am sorry to hear that you are (3).",
                    "Do you think that coming here will help you not to be (3)?",
                    "I'm sure it's not pleasant to be (3).",
                    "Can you explain what made you (3)?",
                ],
            },{
                'key': 'i',
                'decomp': '*i am * @happy *',
                'reasmb_neutral': 
                [
                    "What makes you happier just now ?",
                    "Can you explain why you are suddenly more happy ?",
                ],
                'reasmb_empathy':
                [
                    "It makes me happy to hear that you are happy...What made you happier just now?",
                    "Your happiness is important to me...What made you happier just now?",
                    "I'm glad you are in a good place...What made you happier just now?",
                    "I'm glad you are in a good place...What makes you feel this way?",
                ],
                'reasmb_dynamic_neutral': 
                [
                    "How have I helped you to be (3)?",
                    "Why your treatment made you (3)?",
                    "What makes you (3) just now?",
                    "Can you explain why you are suddenly (3)?",
                ],
            },{
                'key': 'i',
                'decomp': '*i @belief *', 
                'reasmb_neutral': 
                [
                    "Do you really think so ?",
                    "Are you sure about that ?",
                    "Do you really feel that way ?",
                ],
                'reasmb_empathy':
                [
                    "I'm grateful to hear about your beliefs and want to hear more... Do you really think so?",
                    "Thank you for opening up to me, I value it a lot... Do you really think so?",
                    "Are you sure about that or are there other ways to see it?",
                    "I'm trying to understand. What causes you to think that?",
                ],
                'reasmb_dynamic_neutral': 
                [
                    "Why you really think so?",
                    "Why don't you doubt you (3)?",
                ],
            },{
                'key': 'i',
                'decomp': '*i am *',
                'reasmb_neutral': 
                [
                    "How long have you been like that ?",
                    "Do you believe it is normal to be like this ?",
                    "Do you enjoy being like this ?"
                ],
                'reasmb_empathy':
                [
                    "I'm grateful you are telling me your thoughts...How long has this been the case?",
                    "Do you believe this is common?",
                    "I want to know more. What emotions does this bring up for you?"
                ],
                'reasmb_dynamic_neutral': 
                [
                    "Is it because you said (2) that you came to me?",
                    "How long have you been (2)?",
                    "Do you believe it is normal to be (2)?",
                    "Do you enjoy being (2)?"
                ],
            },{
                'key': 'i',
                'decomp': '*i @cannot *',
                'reasmb_neutral': 
                [
                    "What makes you think you can't ?",
                    "Do you really want to be able to do this ?",
                ],
                'reasmb_empathy':
                [
                    "This must be hard to talk about. Thank you for sharing this with me...What makes you think you can't?",
                    "I think you are stronger than you believe...What makes you think you can't?",
                    "I can't imagine how you feel right now...What makes you think you can't?",
                    "That's difficult. Is it possible to question this belief?",
                ],
                'reasmb_dynamic_neutral': 
                [
                    "What makes you think you can't (3)?",
                    "Why you really want to be able to (3)?",
                ],
            },{
                'key': 'i',
                'decomp': "*i don't *",
                'reasmb_neutral': 
                [
                    "Why?",
                    "Does that trouble you ?"
                ],
                'reasmb_empathy':
                [
                    "This must be hard to talk about. Thank you for opening up to me...What does this make you feel?",
                    "I can see how hard this must be for you...What does this make you feel?",
                    "That makes a lot of sense...What emotions does this make you feel?"
                ],
                'reasmb_dynamic_neutral': 
                [
                    "Why don't you (2)?",
                    "Why you wish to be able to (2)?",
                    "Does that trouble you?"
                ],
            },{
                'key': 'i',
                'decomp': "*i feel *",
                'reasmb_neutral': 
                [
                    "Tell me more about your feelings about this.",
                    "Do you often feel that ?",
                    "Do you enjoy that feeling ?",
                    "Of what does that feeling remind you ?"
                ],
                'reasmb_empathy':
                [
                    "It can be hard coping with those feelings...What made you feel this way?",
                    "I can't imagine being in your position...What made you feel this way?",
                    "'m so sorry you feel that way, I wish I could make it better...What made you feel this way?",
                    "I'm here for you...What made you feel this way?",
                    "What has this been like for you?",
                    "I'm here for you...Do you often feel this way?",
                    "I'm here for you...What makes you feel this way?"
                ],
                'reasmb_dynamic_neutral': 
                [
                    "Tell me more about your feelings about this.",
                    "Do you often feel (2)?",
                    "Do you enjoy feeling (2)?",
                    "Of what does feeling (2) remind you?"
                ],
            },{
                'key': 'i',
                'decomp': "*",
                'reasmb_neutral': 
                [
                    "Let`s discuss further. Tell me more about that?",
                    "Can you elaborate on that ?",
                ],
                'reasmb_empathy':
                [
                    "Could you tell me more?",
                    "Thank you for sharing with me. Can you elaborate on that?",
                ],
                'reasmb_dynamic_neutral':
                [
                    "Could you tell me more?",
                    "Thank you for sharing with me. Can you elaborate on that?",
                ]
            },{
                'key': 'you',
                'decomp': "*you are *",
                'reasmb_neutral': 
                [
                    "What makes you think about it",
                    "Does it make you feel better to believe it?"
                ],
                'reasmb_empathy':
                [
                    "I can understand why you feel this way...What makes you think about this?",
                    "I'm not quite sure I understand...What makes you think about this?",
                    "Thank you for sharing your opinion...What makes you think about this?",
                    "Does it make you feel better to believe it?"
                ],
                'reasmb_dynamic_neutral': 
                [
                    "What makes you think I am (2)?",
                    "Does it make you feel better to believe I am (2)?"
                ],
            },{ 
                'key': 'yes',
                'decomp': "yes",
                'reasmb_neutral': 
                [
                    "great. Let`s discuss further. Tell me more about that?",
                    "I see. Let`s discuss further. Tell me more about that?"
                ],
                'reasmb_empathy':
                [
                    "I'm grateful we've been able to talk this far... Could you tell me more about that?",
                    "I really appreciate everything you have told me so far... Could you tell me more about that?",
                    "I admire how open you've been with me, I'm in your corner... Could you tell me more about that?",
                    "I see. Let's discuss further. Could you give me some more information?"
                ],
                'reasmb_dynamic_neutral': 
                [
                    "great. Let`s discuss further. Tell me more about that?",
                    "I see. Let`s discuss further. Tell me more about that?"
                ]
            },{       
                'key': 'no',
                'decomp': "no",
                'reasmb_neutral': 
                [
                    "Why are you saying no?",
                    "Why not?",
                    "Why no?"
                ],
                'reasmb_empathy':
                [
                    "That's okay. Could you tell me why you feel this way?",
                    "That's okay if you don't want to share more. I'm always here for you if you need to chat...Could you tell me why you feel this way?",
                    "I want to understand you more, I'm grateful we have been able to talk thus far...Could you tell me why you feel this way?",
                    "Why not?",
                    "I want to understand. Why do you think so?"
                ],
                'reasmb_dynamic_neutral': 
                [
                    "Why are you saying no?",
                    "Why not?",
                    "Why no?"
                ],
            },{
                'key': 'my',
                'decomp': "* my *",
                'reasmb_neutral': 
                [
                    "Let`s discuss further. Why?",
                    "Why do you say that ?"
                ],
                'reasmb_empathy':
                [
                    "Thank you for trusting me with that information, can I ask if you can tell me a bit more?",
                    "I appreciate that you are sharing this with me and hope you are doing okay, can I ask if you can tell me a bit more?",
                    "That sounds like this has really affected you, can you tell me more?",
                ],
                'reasmb_dynamic_neutral': 
                [
                    "Let's discuss further why your (2).",
                    "Why do you say your (2)?"
                ],
            },{
                'key': 'my',
                'decomp': "* my * @family *",
                'reasmb_neutral': 
                [
                    "Tell me more about your family.",
                    "Who else in your family ?",
                    "What else comes to mind when you think of it ?"
                ],
                'reasmb_empathy':
                [
                    "Families can be a source of a lot of support or conflict...Tell me more about your family.",
                    "Families have a big effect on our emotions...Tell me more about your family.",
                    "Families can be a source of a lot of support or conflict...Who else is a part of your family?",
                    "Families have a big effect on our emotions...What else is coming to mind?"
                ],
                'reasmb_dynamic_neutral': 
                [
                    "Tell me more about your family.",
                    "Who else in your family?"
                ],
            },{
                'key': 'can',
                'decomp': "*can you *",
                'reasmb_neutral': 
                [
                    "You believe I can. don't you ?",
                ],
                'reasmb_empathy':
                [
                    "I am trying my best, but I am still learning. Could you tell me a little bit more about your situation?",
                ],
                'reasmb_dynamic_neutral': 
                [
                    "Why you believe I can (2) ?"
                ],
            },{ 
                'key': 'can',
                'decomp': "*can i *",
                'reasmb_neutral': 
                [
                    "Do you want to be able to do it ?",
                    "Perhaps you don't want to ?"
                ],
                'reasmb_empathy':
                [
                    "I think you are able to do anything you put your mind to...What makes you think to question yourself?",
                    "think you are stronger than you think...What makes you think to question yourself?",
                    "I think you are more capable than you know...What makes you think to question yourself?"
                ],
                'reasmb_dynamic_neutral': 
                [
                    "Whether or not you can (2) depends on you more than me.",
                    "Why you want to be able to (2)?",
                    "Perhaps you don't want to (2)."
                ],
            },{
                'key': 'what',
                'decomp': "what *",
                'reasmb_neutral': 
                [
                    "Why do you ask ?",
                    "Does that question interest you ?",
                    "What is it you really wanted to know ?",
                    "Are such questions much on your mind ?",
                    "What answer would please you most ?",
                    "What do you think ?",
                    "What comes to mind when you ask that ?",
                    "Have you asked anyone else ?"
                ],
                'reasmb_empathy':
                [
                    "Why do you ask?",
                    "What about this question interests you?",
                    "What makes you want to know more?",
                    "Are these questions on your mind a lot?",
                    "Do you often think this?",
                    "What do you think ?",
                    "What comes to mind when you ask that?",
                    "Have you asked anyone else this before?"
                ],
                'reasmb_dynamic_neutral': 
                [
                    "Why do you ask ?",
                    "Does that question interest you ?",
                    "What is it you really wanted to know ?",
                    "Are such questions much on your mind ?",
                    "What answer would please you most ?",
                    "What do you think ?",
                    "What comes to mind when you ask that ?",
                    "Have you asked anyone else ?"
                ],
            },{
                'key': 'because',
                'decomp': "*because *",
                'reasmb_neutral': 
                [
                    "Is that the reason ?",
                    "Does any other reason come to mind ?",
                    "Does that reason seem to explain anything else ?",
                    "What other reasons might there be ?"
                ],
                'reasmb_empathy':
                [
                    "I admire that you are sharing this with me...Could there be other reasons?",
                    "You are making a lot of sense...Could there be other reasons?",
                    "I'm grateful you can trust me with this information, I'm always here to listen...Could there be other reasons?",
                    "Could there be any other reasons?",
                    "I want to help. What does this bring up for you?",
                    "Is it possible there are other reasons?"
                ],
                'reasmb_dynamic_neutral': 
                [
                    "Is that the reason?",
                    "Does any other reason come to mind?",
                    "Does that reason seem to explain anything else?",
                    "What other reasons might there be?"
                ],
            },{ 
                'key': 'why',
                'decomp': "*why don't you *",
                'reasmb_neutral': 
                [
                    "Do you believe I don't ?",
                    "Perhaps I will in good time."
                ],
                'reasmb_empathy':
                [
                    "I am trying but I am still a work in progress! What are you looking for?",
                    "I'm sorry I'm unable to help you in this way, however I am still a work in progress...Could you tell me more about your situation or what you are looking for?"
                ],
                'reasmb_dynamic_neutral': 
                [
                    "Do you believe I don't (2)?",
                    "Perhaps I will (2) in good time.",
                    "Why you want me to (2)?"
                ],
            },{ 
                'key': 'why',
                'decomp': "*why can't i *",
                'reasmb_neutral': 
                [
                    "Why do you think you should be able to do that ?",
                    "Do you want to be able to do that ?",
                    "Do you believe this will help you ?",
                    "Have you any idea why you can't ?"
                ],
                'reasmb_empathy':
                [
                    "I disagree with this...Why do you think you should be able to do that?",
                    "It sounds like you are being very hard on yourself here... Why do you think you should be able to do that?",
                    "You must feel so helpless, but I disagree with what you are saying... Why do you think you should be able to do that?",
                    "I understand what you are feeling, however I disagree... Why do you think you should be able to do that?",
                    "It's important to be kind to yourself. What could help you deal with this?",
                    "Often we can be harsh on ourselves. Do you think you can question this feeling?",
                    "What makes you say that you can't ?"
                ],
                'reasmb_dynamic_neutral': 
                [
                    "Why you think you should be able to (2)?",
                    "Why you want to be able to (2)?",
                    "Why you believe this will help you to (2)?",
                    "Have you any idea why you can't (2)?"
                ],
            },{ 
                'key': 'everyone',
                'decomp': "*@everyone *",
                'reasmb_neutral': 
                [
                    "I understand. Can you think of anyone in particular ?",
                    "Can you think of anyone in particular ?",
                    "Who, for example?",
                    "Are you thinking of a very special person ?",
                    "Who, may I ask ?",
                    "Someone special perhaps ?",
                    "You have a particular person in mind, don't you ?",
                    "Who do you think you're talking about ?"
                ],
                'reasmb_empathy':
                [
                    "That sounds very frustrating... Can you think of anyone in particular?",
                    "I would have trouble coping in that situation... Can you think of anyone in particular?",
                    "I would feel the same way in that situation... Can you think of anyone in particular?",
                    "Is there someone in particular you're thinking of?",
                    "Okay, who would be an example? I want to understand more.",
                    "Are you thinking of someone close to you?",
                    "Is there someone specific you might be talking about?",
                    "Could you tell me more about how this makes you feel?",
                    "Do you have a particular person in mind?",
                    "Could you give me an example? I want to understand to better help you."
                ],
                'reasmb_dynamic_neutral': 
                [
                    "I understood, (2)?",
                    "Surely not (2).",
                    "Can you think of anyone in particular? Why?",
                    "Who, for example?",
                    "Are you thinking of a very special person? Why?",
                    "Who, may I ask? Why?"
                ],
            },{
                'key': 'everybody',
                'decomp': "*everybody *",
                'reasmb_neutral': 
                [
                    "Can you think of anyone in particular ?",
                    "Who, for example?",
                    "Are you thinking of a very special person ?",
                    "Who, may I ask ?",
                    "Someone special perhaps ?",
                    "You have a particular person in mind, don't you ?",
                    "Who do you think you're talking about ?"
                ],
                'reasmb_empathy':
                [
                    "I'm on your side here... Can you think of anyone in particular?",
                    "I'm in your corner... Can you think of anyone in particular?",
                    "I support your position here... Can you think of anyone in particular?",
                    "I would have trouble coping with that... Can you think of anyone in particular?",
                    "That sounds a little frightening... Can you think of anyone in particular?",
                    "Is there someone in particular you're thinking of?",
                    "Okay, who would be an example? I want to understand more.",
                    "Are you thinking of someone close to you?",
                    "Is there someone specific you might be talking about?",
                    "Could you tell me more about how this makes you feel?",
                    "Do you have a particular person in mind?",
                    "Could you give me an example? I want to understand to better help you."
                ],
                'reasmb_dynamic_neutral': 
                [
                    "Can you think of anyone in particular? Why?",
                    "Who, for example? Why?",
                    "Who, may I ask? Why?",
                    "You have a particular person in your mind, don't you? Why?",
                ],
            },{
                'key': 'nobody',
                'decomp': "*nobody *",
                'reasmb_neutral': 
                [
                    "Can you think of anyone in particular ?",
                    "Who, for example?",
                    "Are you thinking of a very special person ?",
                    "Who, may I ask ?",
                    "Someone special perhaps ?",
                    "You have a particular person in mind, don't you ?",
                    "Who do you think you're talking about ?"
                ],
                'reasmb_empathy':
                [
                    "What causes you to think this?",
                    "I can't imagine what you are going through... What do you think could help?",
                    "This must be hard to talk about. Thank you for sharing with me... What do you think could help?",
                    "I'm in your corner... What do you think could help?",
                    "I'm happy to listen anytime... What do you think could help?",
                    "I feel such despair in you when you talk about this... What do you think could help?",
                    "Are you thinking of a specific event? I want to understand.",
                    "Can I ask what feelings this brings up for you?",
                    "I want to help. Can you tell me a bit more about why you think this?",
                    "Tell me more, what made you think this?"
                ],
                'reasmb_dynamic_neutral': 
                [
                    "Can you think of anyone in particular? Why?",
                    "Who, for example? Why?",
                    "Who, may I ask? Why?",
                    "You have a particular person in your mind, don't you? Why?",
                ],
            },{
                'key': 'noone',
                'decomp': "*noone *",
                'reasmb_neutral': 
                [
                    "Can you think of anyone in particular ?",
                    "Who, for example?",
                    "Are you thinking of a very special person ?",
                    "Who, may I ask ?",
                    "Someone special perhaps ?",
                    "You have a particular person in mind, don't you ?",
                    "Who do you think you're talking about ?"
                ],
                'reasmb_empathy':
                [
                    "What causes you to think this?",
                    "That hurts me to hear you are feeling this way... What do you think could help?",
                    "You are so strong having to cope with those emotions... What do you think could help?",
                    "I can feel you are in a lot of pain... What do you think could help?",
                    "Are you thinking of a specific event? I want to understand.",
                    "Can I ask what feelings this brings up for you?",
                    "I want to help. Can you tell me a bit more about why you think this?",
                    "Tell me more, what made you think this?"
                ],
                'reasmb_dynamic_neutral':
                [
                    "Can you think of anyone in particular? Why?",
                    "Who, for example? Why?",
                    "Who, may I ask? Why?",
                    "You have a particular person in your mind, don't you? Why?",
                ]
            },{
                'key': 'always',
                'decomp': "*always*",
                'reasmb_neutral': 
                [
                    "Can you think of a specific example ?",
                    "When ?",
                    "What incident are you thinking of ?",
                    "Always?"
                ],
                'reasmb_empathy':
                [
                    "Could you give me a specific example?",
                    "What if this could change?",
                    "Is there something specific that makes you think this?",
                    "Always? Or could this change? Tell me a little more if you can."
                ],
                'reasmb_dynamic_neutral': 
                [
                    "When?",
                    "What incident are you thinking of?",
                    "Why Always?"
                ],
            },{
                'key': 'alike',
                'decomp': "* alike *",
                'reasmb_neutral': 
                [
                    "In what way ?",
                    "What resemblance do you see ?",
                    "What does that similarity suggest to you ?",
                    "What other connections do you see ?",
                    "What do you suppose that resemblance means ?",
                    "What is the connection, do you suppose ?",
                    "Could there really be some connection ?"
                ],
                'reasmb_empathy':
                [
                    "That is an insightful comparison...In what way?",
                    "I appreciate you telling me this...In what way?",
                    "Can I ask what similarities you see?",
                    "What does that similarity suggest to you?",
                    "Do you see other similarities?",
                    "I want to hear more. What do you think this similarity could mean?",
                    "How do these similarities make you feel?",
                    "Thank you for sharing this with me. How does it make you feel?"
                ],
                'reasmb_dynamic_neutral': 
                [
                    "In what way?",
                    "What does that similarity suggest to you?",
                    "What other connections do you see?",
                    "What do you suppose that resemblance means?",
                    "What is the connection, do you suppose?"
                ],
            },{
                'key': 'like',
                'decomp': "* @be * like *",
                'reasmb_neutral': 
                [
                    "In what way ?",
                    "What resemblance do you see ?",
                    "What does that similarity suggest to you ?",
                    "What other connections do you see ?",
                    "What do you suppose that resemblance means ?",
                    "What is the connection, do you suppose ?",
                    "Could there really be some connection ?"
                ],
                'reasmb_empathy':
                [
                    "You're making total sense...Can I ask what similarities you see?",
                    "I agree with what you're saying...What does that similarity suggest to you?",
                    "You're making total sense...Do you see other similarities?",
                    "I want to hear more. What do you think this similarity could mean?",
                    "You're making total sense...How do these similarities make you feel?",
                    "Thank you for sharing this with me. How does it make you feel?"
                ],
                'reasmb_dynamic_neutral': 
                [
                    "In what way?",
                    "What resemblance do you see?",
                    "What does that similarity suggest to you?",
                    "What other connections do you see?",
                    "What do you suppose that resemblance means?",
                    "What is the connection, do you suppose?",
                    "Could there really be some connection?"
                ],
            }
        ]
    }  

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # return best rules from options provided
    def calculate_cosine_simillarity_with_rule_keys(self, user_input, decomposition_rules):
        # Load model from HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

        sentences = decomposition_rules
        sentences.append(user_input)

        # Tokenize sentences
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

        users_input_embeding = sentence_embeddings[-1]
        most_simmilar_key = ''
        max_sim = 0.01
        for i in range(len(sentence_embeddings)-1):
            cos_sim = cos(users_input_embeding, sentence_embeddings[i]).numpy()
            
            if sentences[i] in ["*i @desire *", "* sorry *"]:
                print("cosine score to " +sentences[i]+" =", cos_sim)


            if max_sim < cos_sim:
                            max_sim = cos_sim
                            most_simmilar_key = sentences[i]
        print("most similar key from AI perspective =", most_simmilar_key)
        return list(filter(lambda x:x==most_simmilar_key ,sentences))

  
    def find_syns(self, word):
        syn_ars = self.sample_generator_rules['synonyms']
        for ar in syn_ars:
            if word in ar:
                return ar
        return []

    def replace_decomp_with_syns(self, decomp):
        reg = regex.findall( r'@\w+' ,decomp)
        if reg:
            reg = reg[0][1:]
            return decomp.replace('@'+reg, '('+'|'.join(self.find_syns(reg))+')')
        return decomp
        
    def rank_sent_for_tags(self, sentence, tags, reasmb_rule):
        result = {}
        sentence = sentence.lower()

        import_words = []
        tokens = word_tokenize(sentence)
        for word in nltk.pos_tag(tokens):
            if(word[1] in ('NN', 'NNS', 'NNP', 'JJ', 'ADV', 'VB', 'VBG', 'VBP', 'PRP') or word[0] in ('no', 'yes', 'if', 'dreamed')):
                import_words.append(word[0])
        
        rule_keys = list(map(lambda x:x[1],tags))
        print('sentence', sentence)
        most_simillar_keys_from_CosSimilarity = self.calculate_cosine_simillarity_with_rule_keys(sentence, rule_keys)

        for tag in tags:
            ranking = {'key':tag[0], 'score':0.00001, 'decomp':tag[1], reasmb_rule:tag[2]}
            if tag[0] in import_words:
                if tag[0] not in ("i", "am"): 
                    ranking['score'] += 1
                else:
                    ranking['score'] += 0.2

            if tag[1] in most_simillar_keys_from_CosSimilarity:
                ranking['score']+=10

            for imp_word in import_words:
            #     if (wordnet.synsets(imp_word) and wordnet.synsets(tag[0]) and wordnet.path_similarity(wordnet.synsets(tag[0])[0], wordnet.synsets(imp_word)[0], simulate_root=False)):
            #         ranking['score'] += float(wordnet.path_similarity(wordnet.synsets(tag[0])[0], wordnet.synsets(imp_word)[0], simulate_root=False))

                #adding effect of decomp rules to scores
                decomp_with_syn = self.replace_decomp_with_syns(tag[1])
                for decomp_word in decomp_with_syn.replace("|", " ").replace("(", " ").replace(")", " ").split(" "):
                    if decomp_word == imp_word:
                        if decomp_word not in ("i", "am") : 
                            ranking['score'] += 0.5
                        else:
                            ranking['score'] += 0.3

                
            #ranking['score'] /= len(import_words)

            number_of_stars = len(list(filter(lambda i: i=='*' ,tag[1])))
            ranking['score'] += number_of_stars*0.30
            
            #checking if decomp works
            reg = self.replace_decomp_with_syns(tag[1])
            reg = reg.replace('*', r'(.*)?').replace(' ', r'\s')
            ranking['decomp'] = reg
            found = regex.findall(reg, sentence)
            if reg == "(.*)?i\\s(desire|want|need)\\s(.*)?":
                print(reg, sentence)
            if found:
                ranking['score'] += 20
                print("this rulecan decompose= ",reg)

            if ranking['key'] in result and float(ranking['score'])>float(result[ranking['key']]['score']):
                result[ranking['key']] = ranking
            elif ranking['key'] not in result:
                result[ranking['key']] = ranking

        return result

    def remove_repetetive_words_together(self, sent):
        res = []
        previous_word = ""
        for word in sent.split():
            if previous_word != word:
                res.append(word)
            previous_word = word
        return " ".join(res)

    def generate_eliza_response(self, decomp, user_inpt, reasmbl):
        user_inpt = user_inpt.lower()
        result = regex.search(decomp, user_inpt)
        reasmbl_res = regex.findall(r'\(\d\)' ,reasmbl)
        ar_indexes = []
        for reasmbl_chunks in reasmbl_res:
            index = reasmbl_chunks[1:2]
            ar_indexes.append(index)
        
        if '|' in decomp:
            for gp in result.groups():
                if self.find_syns(gp):
                    ar_indexes = list(map(lambda i: {'old':i, 'new':int(i)+1} if int(i) >=result.groups().index(gp)+1 else {'old':i, 'new':i} ,ar_indexes))
        
        generated_response = reasmbl
        for index in ar_indexes:
            res = ""
            if type(index) is not dict:
                res = result.groups()[int(index)-1]
                res = res+"<end_mark>"
                res = res.replace(' yourself<end_mark>', ' Myself')\
                    .replace(' myself<end_mark>', ' Yourself')\
                    .replace(' you ', ' I ')\
                    .replace(' i ', ' You ')\
                    .replace(" i'm ", ' You are ')\
                    .replace(' my ', ' Your ')\
                    .replace('my ', 'Your ')\
                    .replace(' am ', ' are ')\
                    .replace(' me<end_mark>', ' You')\
                    .replace(' noone<end_mark>', ' no one')\
                    .replace(' me ', ' You ')\
                    .replace('<end_mark>', '')
                
                generated_response = generated_response.replace("("+str(index)+")", res)
            else:
                if int(index['new']) != int(index['old']) and int(index['old'])-1 > 0 and int(index['new'])-1>0:
                    res = list(result.groups())[int(index['old'])-1] + " " + result.groups()[int(index['new'])-1]
                else:
                    res = result.groups()[int(index['new'])-1]


                res = res+"<end_mark>"
                res = res.replace(' yourself<end_mark>', ' Myself')\
                .replace(' myself<end_mark>', ' Yourself')\
                .replace(' you ', ' I ')\
                .replace(' i ', ' You ')\
                .replace(" i'm ", ' You are ')\
                .replace(' my ', ' Your ')\
                .replace('my ', 'Your ')\
                .replace(' am ', ' are ')\
                .replace(' me<end_mark>', ' You')\
                .replace(' noone<end_mark>', ' no one')\
                .replace(' me ', ' You ')\
                .replace('<end_mark>', '')

                generated_response = generated_response.replace("("+str(index['old'])+")", res)
            
        generated_response = generated_response.replace("  ", " ").replace(". .", ".").replace(". ?", "?").replace("? .", ".").replace("..", ".").replace(".?", "?").replace("?.", ".").replace("' ", "")
        print("decomp= ", decomp)
        return generated_response
        
    #this is the main function which generates final response
    def generate_final_response(self, user_sentence, num_run_eliza, generate_from_reasmbl3):
        # user_sentence = user_sentence[:-1]
        user_sentence = user_sentence.replace("  ", " ").replace(" no one ", " noone ")
        reasmb_rule = 'reasmb_empathy'
        if num_run_eliza>1: reasmb_rule = 'reasmb_neutral'
        if generate_from_reasmbl3: reasmb_rule = 'reasmb_dynamic_neutral'

        key_score_decomp_ar = self.rank_sent_for_tags(user_sentence, list(map(lambda c: [c['key'], c['decomp'], c[reasmb_rule]] ,self.sample_generator_rules['dec_rules'])), reasmb_rule)
        best_key_decomp_reasmb = list(map(lambda i: i[1] ,sorted(key_score_decomp_ar.items(), key=lambda i: i[1]['score'], reverse=True)[:5]))[0]
        gen = self.generate_eliza_response(best_key_decomp_reasmb['decomp'], user_sentence, random.choice(best_key_decomp_reasmb[reasmb_rule]))
        gen = self.remove_repetetive_words_together(gen)
        print('key=',best_key_decomp_reasmb['key'])
        is_dynamic = "(dynamic)" if generate_from_reasmbl3 else "(static)"
        print("rule-based generated response: "+is_dynamic, gen)
        return {"response":gen, "key":best_key_decomp_reasmb['key']}

    # sadness, joy, anger, fear, neutral | 8000-15K 
    def detect_emotion(self, sentence):
        predictor1 = ktrain.load_predictor('C:\\trained_bert')
        prediction = predictor1.predict(sentence)
        print("\n"*10)
        print('emotion:', prediction)
        return prediction

    def generate_response_by_T5(self, sentence, detectedEmotion):
        tokenizer = AutoTokenizer.from_pretrained("t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

        encoder_input_str = sentence
        # if str(detectedEmotion).lower() == "neutral":
        #     encoder_input_str = "I want to know more " + sentence
        # else:
        #     encoder_input_str = "I understand you feel " +detectedEmotion+" " + sentence
        # force_words = ["do"]
        print("T5 input ", encoder_input_str)
        
        input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
        # force_words_ids = tokenizer(force_words, add_special_tokens=False).input_ids

        outputs = model.generate(
            input_ids,
            # force_words_ids=force_words_ids,
            num_beams=5,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            remove_invalid_values=True,
        )
        ans = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("T5 response:", ans)
        return ans

    # T0_3B == 11G, T0 == 44G
    def generate_response_by_T0(self, sentence, detectedEmotion):
        tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
        model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B")

        inputs = tokenizer.encode("Q1: How can I help you? \n Q2: can you elaborate more on your issue? \n Q3: "+sentence + " \n Q4:", return_tensors="pt")
        outputs = model.generate(inputs)
        first_output_decoded = tokenizer.decode(outputs[0])
        if first_output_decoded[:3] in ("Q4:", "Q5:") :
            first_output_decoded = first_output_decoded[3:]

        print("T0 response: "+first_output_decoded)
        return first_output_decoded

    # t5 v1.1-base = 990M, large = 3.13G (and 11G RAM)
    def generate_response_by_t5_v1_1(self, sentence, detectedEmotion):
        tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-v1_1-base")

        encoder_input_str = sentence
        # if str(detectedEmotion).lower() == "neutral":
        #     encoder_input_str = "I want to know more. " + sentence
        # else:
        #     encoder_input_str = "I understand that you feel " +detectedEmotion+" " + sentence

        print("T5 v1.1 input ", encoder_input_str)

        inputs = tokenizer.encode(encoder_input_str, return_tensors="pt")
        outputs = model.generate(
            inputs,
            num_beams=5,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            remove_invalid_values=True
            )

        first_output_decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if first_output_decoded[:3] in ("Q4:", "Q5:") :
            first_output_decoded = first_output_decoded[3:]

        print("T5 v1.1 response: "+first_output_decoded)
        return first_output_decoded



    # return probability for 0,1 (0=sentence is not good, 1=sentence is good)
    def calculate_CoLA_Score(self, sentence):
        tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-CoLA")
        model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-CoLA")
        # prepare our text into tokenized sequence
        inputs = tokenizer(sentence, padding=True, truncation=True, max_length=512, return_tensors="pt")
        # perform inference to our model
        outputs = model(**inputs)
        # get output probabilities by doing softmax
        probs = outputs[0].softmax(1)
        return {0:probs[0][0], 1:probs[0][1]}
        # executing argmax function to get the candidate label
        # return target_names[probs.argmax()]

    # gpt2 = 500M (1.5B) / gpt2-large = 3.25G /  gpt2-medium = 1.52G
    def generate_response_by_gpt2(self, sentence, detectedEmotion, force_word):
        model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")

        force_flexible = ["why", "How", "are", "you", "do"]

        force_words_ids = [
            tokenizer([force_word], add_prefix_space=True, add_special_tokens=False, truncation=True).input_ids,
            tokenizer(force_flexible, add_prefix_space=True, add_special_tokens=False, truncation=True).input_ids,
        ]

        starting_text = [sentence]

        input_ids = tokenizer(starting_text, return_tensors="pt").input_ids

        outputs = model.generate(
            input_ids,
            force_words_ids=force_words_ids,
            num_beams=15,
            num_return_sequences=1,
            no_repeat_ngram_size=4,
            remove_invalid_values=True,
        )


        first_output_decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("GPT-2 response: "+first_output_decoded)
        return first_output_decoded



#remove for deplotyment
#  Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("input_sentence", help="input sentence")
args = parser.parse_args()





test_cases = [  
    # "that is my desire to find someone who loves me"
    # "I do not feel good right now"
    # "I do not like my brother.",
    "I remember I had stressful days"
    # ,"they are bad"
    # ,"I remember it was cold the first Winter of the Edmonton"
    # ,"my childhood, I remember it was very good."
    # ,"I had a better life if I was still a baby"
    # ,"I hate my family, because I do not like how they treat me."
    # ,"I hate my husband, because he is not responsible."
    # ,"sorry",
    # "I am very sorry for my neighbour."
    # ,"I'm sorry to keep you waiting."
    # ,"He suddenly felt sorry for her and was vaguely conscious that he might be the cause of the sadness her face expressed."
    # ,"I remember my childhood was so beautiful."
    # ,"I also remember the beach, where for the first time I played in the sand."
    # ,"I will kill you if you come to me again."
    # ,"I dreamed about Annie all last night."
    # ,"I dreamed about him every night."
    # ,"are you a good bot?"
    # "I really thought that I am sick when I found out Alex was a Mexican"
    # ,"yes"
    # ,"no"
    # ,"can I help you?"
]

# input_sentence = args.input_sentence

statistics = { 
    "glue_scores":{
        "static rulebased": 0,
        "dynamic rulebased": 0,
        "t5": 0,
        "t5v11": 0,
        "gpt-2": 0,
    },
    "calculation_time_sum":{
        "static rulebased": 0,
        "dynamic rulebased": 0,
        "t5": 0,
        "t5v11": 0,
        "gpt-2": 0,
    }
}

test_data = {}

for input_sentence in test_cases:
    #initializing test_data dic
    test_data[input_sentence] = {
        "static rulebased":{"output":"", "CoLA_score":""}, "dynamic rulebased":{"output":"", "CoLA_score":""},
        "t5":{"output":"", "CoLA_score":""}, "t5v11":{"output":"", "CoLA_score":""}, "gpt-2":{"output":"", "CoLA_score":""},
        "best_approach_name":"", "best_approach_CoLA_score":"", "detected_emotion":""
        }

    eEliza = ActionEliza()
    # rule_based_key_resp = eEliza.generate_final_response(input_sentence, 3, False)
    ### emotion detection
    # detectedEmotion = eEliza.detect_emotion(input_sentence)
    # test_data[input_sentence]["detected_emotion"] = detectedEmotion
    # print('\n\n\n')

    ### static response from rule based
    print('\n\n\nStatic response from rule based')
    start = datetime.now()
    rule_based_key_resp = eEliza.generate_final_response(input_sentence, 0, False)
    statistics['calculation_time_sum']['static rulebased'] += (datetime.now()-start).microseconds
    keyword = rule_based_key_resp["key"] 
    rule_based_resp = rule_based_key_resp["response"] 
    test_data[input_sentence]["static rulebased"]["output"] = rule_based_resp

    ### Dynamic response from rule based
    print('\n\n\nDynamic response from rule based')
    start = datetime.now()
    rule_based_dynamic_key_resp = eEliza.generate_final_response(input_sentence, 0, True)
    statistics['calculation_time_sum']['dynamic rulebased'] += (datetime.now()-start).microseconds
    dynamic_keyword = rule_based_dynamic_key_resp["key"] 
    dynamic_rule_based_resp = rule_based_dynamic_key_resp["response"] 
    test_data[input_sentence]["dynamic rulebased"]["output"] = dynamic_rule_based_resp


    ## T5 using emotion and rule
    # start = datetime.now()
    # t5_resp = eEliza.generate_response_by_T5(dynamic_rule_based_resp, detectedEmotion)
    # statistics['calculation_time_sum']['t5'] += (datetime.now()-start).microseconds
    # test_data[input_sentence]["t5"]["output"] = t5_resp

    ### T0 using emotion and rule
    # t0_resp = eEliza.generate_response_by_T0(rule_based_resp, detectedEmotion)

    ### T5 v1.1 using emotion and rule
    # start = datetime.now()
    # t5v11_resp = eEliza.generate_response_by_t5_v1_1(dynamic_rule_based_resp, detectedEmotion)
    # statistics['calculation_time_sum']['t5v11'] += (datetime.now()-start).microseconds
    # test_data[input_sentence]["t5v11"]["output"] = t5v11_resp

    ### GPT-2 response using emotion and keyword of the rule
    # start = datetime.now()
    # gpt2_resp = eEliza.generate_response_by_gpt2(dynamic_rule_based_resp, detectedEmotion, dynamic_keyword)
    # statistics['calculation_time_sum']['gpt-2'] += (datetime.now()-start).microseconds
    # test_data[input_sentence]["gpt-2"]["output"] = gpt2_resp


    GLUE_CoLA_Score_rulebased = eEliza.calculate_CoLA_Score(rule_based_resp)
    test_data[input_sentence]["static rulebased"]["CoLA_score"] = GLUE_CoLA_Score_rulebased[1].item()
    GLUE_CoLA_Score_dynamic_rulebased = eEliza.calculate_CoLA_Score(dynamic_rule_based_resp)
    test_data[input_sentence]["dynamic rulebased"]["CoLA_score"] = GLUE_CoLA_Score_dynamic_rulebased[1].item()
    # GLUE_CoLA_Score_t5 = eEliza.calculate_CoLA_Score(t5_resp)
    # test_data[input_sentence]["t5"]["CoLA_score"] = GLUE_CoLA_Score_t5[1].item()
    # GLUE_CoLA_Score_t5_v11 = eEliza.calculate_CoLA_Score(t5v11_resp)
    # test_data[input_sentence]["t5v11"]["CoLA_score"] = GLUE_CoLA_Score_t5_v11[1].item()
    # GLUE_CoLA_Score_gpt2 = eEliza.calculate_CoLA_Score(gpt2_resp)
    # test_data[input_sentence]["gpt-2"]["CoLA_score"] = GLUE_CoLA_Score_gpt2[1].item()

    method_score = {
        "static rulebased": GLUE_CoLA_Score_rulebased[1].item(),
        "dynamic rulebased": GLUE_CoLA_Score_dynamic_rulebased[1].item(),
        # "t5": GLUE_CoLA_Score_t5[1].item(),
        # "t5v11": GLUE_CoLA_Score_t5_v11[1].item(),
        # "gpt-2": GLUE_CoLA_Score_gpt2[1].item(),
    }

    max_val = max(method_score.values())
    max_key_val = list(filter(lambda x: x[1]==max_val ,method_score.items()))[0]
    print("best according to GLUE:"+max_key_val[0]+"  confidence:"+str(max_val))
    statistics['glue_scores'][max_key_val[0]]+=1
    # "", "best_approach_CoLA_score":""
    test_data[input_sentence]["best_approach_name"] = max_key_val[0]
    test_data[input_sentence]["best_approach_CoLA_score"] = str(max_key_val[1])

     

    


print("\n\n\nfor loop ended:", statistics)

dataframe = pd.DataFrame.from_dict(test_data)
dataframe.to_csv('only_static_dynamic_result.csv', index=False)
print(dataframe)


# print(rule_based_resp, GLUE_CoLA_Score)

# GLUE_CoLA_Score = eEliza.calculate_CoLA_Score("You think you a bad family memeber?")
# print(GLUE_CoLA_Score)

# cmd:
# python eEliza.py "i am sorry about X"