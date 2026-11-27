# Error analysis sample (arm D, 25 of 26 misses; categories are PROPOSED)

## 1603.01514:6568a31241167f618ef5ede939053feaa2fb0d7e
**Q:** Do they add one latent variable for each language pair in their Bayesian model?
- answer types: ['yes_no']; evidence paragraphs: 1; gold sections: ['Multilingual Model']; question words found in evidence: 0.5
- categories (PROPOSED): ['none_of_the_proposed']
- **gold evidence:** The multilingual model uses word alignments between sentences in a parallel corpus to exploit role correspondences across languages. We make copies of the monolingual model for each language and add additional crosslingual latent variables (CLVs) to couple the monolingual models, capturing crossling
- top-1 [Multilingual Model] 'This is the first work to incorporate the coupling of aligned arguments directly in a Bayesian SRL model. This makes it easier to see how to extend this model in a principled way to incorporate additi'
- top-2 [Conclusions] 'We propose a Bayesian model of semantic role induction (SRI) that uses crosslingual latent variables to capture role alignments in parallel corpora. The crosslingual latent variables capture correlati'
- top-3 [Multilingual Model] 'Second, having one joint Bayesian model allows us to use the same model in various semi-supervised learning settings, just by fixing the annotated variables during training. Section SECREF29 evaluates'

## 1604.08504:3cd185b7adc835e1c4449eff81222f5fc15c8500
**Q:** How do they detect spammers?
- answer types: ['abstractive']; evidence paragraphs: 4; gold sections: ['Performance Comparisons with Baseline', 'Topic-based Features']; question words found in evidence: 0.0
- categories (PROPOSED): ['vocabulary_mismatch', 'multi_paragraph_evidence', 'evidence_in_unusual_section']
- **gold evidence:** Using the LDA model, each person in the dataset is with a topic probability vector $X_i$ . Assume $x_{ik}\in X_{i}$ denotes the likelihood that the $\emph {i}^{th}$ tweet account favors $\emph {k}^{th}$ topic in the dataset. Our topic based features can be calculated as below.
- **gold evidence:** Global Outlier Standard Score measures the degree that a user's tweet content is related to a certain topic compared to the other users. Specifically, the "GOSS" score of user $i$ on topic $k$ can be calculated as Eq.( 12 ):
- **gold evidence:** Local Outlier Standard Score measures the degree of interest someone shows to a certain topic by considering his own homepage content only. For instance, the "LOSS" score of account $i$ on topic $k$ can be calculated as Eq.( 13 ):
- **gold evidence:** Three baseline classification methods: Support Vector Machines (SVM), Adaboost, and Random Forests are adopted to evaluate our extracted features. We test each classification algorithm with scikit-learn BIBREF9 and run a 10-fold cross validation. On each dataset, the employed classifiers are trained
- top-1 [Observation] 'After exploring the homepages of a substantial number of spammers, we have two observations. 1) social spammers can be divided into two categories. One is content polluters, and their tweets are all a'
- top-2 [Abstract] 'Spammer detection on social network is a challenging problem. The rigid anti-spam rules have resulted in emergence of"smart"spammers. They resemble legitimate users who are difficult to identify. In t'
- top-3 [Topic-based Features] 'Denote $\\emph {k}^{th}$2 our first topic-based feature, and it hopefully can get good performance on spammer detection.'

## 1604.08504:dac087e1328e65ca08f66d8b5307d6624bf3943f
**Q:** LDA is an unsupervised method; is this paper introducing an unsupervised approach to spam detection?
- answer types: ['yes_no']; evidence paragraphs: 1; gold sections: ['Performance Comparisons with Baseline']; question words found in evidence: 0.0
- categories (PROPOSED): ['vocabulary_mismatch']
- **gold evidence:** Three baseline classification methods: Support Vector Machines (SVM), Adaboost, and Random Forests are adopted to evaluate our extracted features. We test each classification algorithm with scikit-learn BIBREF9 and run a 10-fold cross validation. On each dataset, the employed classifiers are trained
- top-1 [Abstract] 'Spammer detection on social network is a challenging problem. The rigid anti-spam rules have resulted in emergence of"smart"spammers. They resemble legitimate users who are difficult to identify. In t'
- top-2 [Conclusion] 'In this paper, we propose a novel feature extraction method to effectively detect "smart" spammers who post seemingly legitimate tweets and are thus difficult to identify by existing spammer classific'
- top-3 [Introduction] 'In this paper, we propose an efficient feature extraction method. In this method, two new topic-based features are extracted and used to discriminate human-like spammers from legitimate users. We cons'

## 1608.01884:1dc2da5078a7e5ea82ccd1c90d81999a922bc9bf
**Q:** Do the authors conduct experiments on the tasks mentioned?
- answer types: ['yes_no']; evidence paragraphs: 1; gold sections: ['Current state of the art']; question words found in evidence: 0.0
- categories (PROPOSED): ['vocabulary_mismatch', 'evidence_in_unusual_section']
- **gold evidence:** What may be more surprising is that currently (July 2016), machine translation programs are unable to choose the feminine plural pronoun even when the only possible antecedent is a group of women. For instance, the sentence “The girls sang a song and they danced” is translated into French as “Les fi
- top-1 [Language-specific issues] 'Looking further ahead, it is certainly possible that gender distinctions will be abandoned in the Romance languages, or even that English will have driven all other languages out of existence, sooner '
- top-2 [Winograd Schemas] 'We will say that an element of a Winograd schema is “solved” if the referent of the pronoun is identified.\n\nAn example of a pair of sentences satisfying conditions 1-3 but not 4 would be\n\nSince women '
- top-3 [Winograd schemas as the basis for challenges for machine translation programs] "A number of the Winograd schemas already published can be “converted” quite easily to this form. Indeed, the above example was constructed in this way; the original form was “Jane knocked on Susan's d"

## 1608.01884:9a7ba5ed1779c664d2cac92494a43517d3e87c96
**Q:** What data do they look at?
- answer types: ['extractive']; evidence paragraphs: 1; gold sections: ['Winograd schemas as the basis for challenges for machine translation programs']; question words found in evidence: 0.0
- categories (PROPOSED): ['vocabulary_mismatch', 'evidence_in_unusual_section']
- **gold evidence:** Care must be taken to avoid relying on, or seeming to rely on, objectionable stereotypes about men and women. One mechanism that can sometimes be used is to include a potentially problematic sentence in both directions. For instance, schema 23 from the WSC collection can be translated into both “The
- top-1 [Winograd schemas as the basis for challenges for machine translation programs] 'In many cases, the identification of the referent of the prounoun in a Winograd schema is critical for finding the correct translation of that pronoun in a different language. Therefore, Winograd sche'
- top-2 [References] 'E. Davis, “Qualitative Spatial Reasoning in Interpreting Text and Narrative” Spatial Cognition and Computation, 13:4, 2013, 264-294.\n\nH. Levesque, E. Davis, and L. Morgenstern, “The Winograd Schema Ch'
- top-3 [Winograd schemas as the basis for challenges for machine translation programs] "A number of the Winograd schemas already published can be “converted” quite easily to this form. Indeed, the above example was constructed in this way; the original form was “Jane knocked on Susan's d"

## 1701.02962:2fbb6322e485e7743ec3fb4bb02d44bf4b5ea8a6
**Q:** What dataset do they use to evaluate their method?
- answer types: ['extractive']; evidence paragraphs: 2; gold sections: ['Dataset', 'Experimental Settings']; question words found in evidence: 0.25
- categories (PROPOSED): ['multi_paragraph_evidence']
- **gold evidence:** For training the models, neural networks require a large amount of training data. We use the existing large-scale antonym and synonym pairs previously used by Nguyen:16. Originally, the data pairs were collected from WordNet BIBREF9 and Wordnik.
- **gold evidence:** We use the English Wikipedia dump from June 2016 as the corpus resource for our methods and baselines. For parsing the corpus, we rely on spaCy. For the lemma embeddings, we rely on the word embeddings of the dLCE model BIBREF10 which is the state-of-the-art vector representation for distinguishing 
- top-1 [Effect of the Distance Feature] 'For evaluation, we make use of the same information regarding dataset and patterns as in Section "Overall Results" , and then replace the distance feature by the direction feature. The results are sho'
- top-2 [Overall Results] 'Table 3 shows the significant performance of our models in comparison to the baselines. Concerning adjectives, the two proposed models significantly outperform the two baselines: The performance of th'
- top-3 [Dataset] 'In order to induce patterns for the word pairs in the dataset, we identify the sentences in the corpus that contain the word pair. Thereafter, we extract all patterns for the word pair. We filter out '

## 1702.03342:2a6003a74d051d0ebbe62e8883533a5f5e55078b
**Q:** which neural embedding model works better?
- answer types: ['abstractive', 'extractive']; evidence paragraphs: 1; gold sections: ['Dataless Classification']; question words found in evidence: 0.4
- categories (PROPOSED): ['table_or_figure_context_also_cited']
- **gold evidence:** Table 3 presents the results of fine-grained dataless classification measured in micro-averaged F1. As we can notice, ESA achieves its peak performance with a few hundred dimensions of the sparse BOC vector. Using our densification mechanism, both the CRC & 3C models achieve equal performance to ESA
- top-1 [Introduction] 'We evaluate our embedding models on two tasks:\n\nThe contributions of this paper are threefold: First, we propose two low cost concept embedding models which requires few hours rather than days to trai'
- top-2 [Related Work] 'Concept/Entity Embeddings: neural embedding models have been proposed to learn distributed representations of concepts/entities. songunsupervised proposed using the popular Word2Vec model BIBREF12 to '
- top-3 [Introduction] 'In this paper we propose two neural embedding models in order to learn continuous concept vectors based on the skip-gram model BIBREF11 . Our first model is the Concept Raw Context model (CRC) which u'

## 1703.07090:c0af8b7bf52dc15e0b33704822c4a34077e09cd1
**Q:** which lstm models did they compare with?
- answer types: ['abstractive']; evidence paragraphs: 1; gold sections: ['Deployment']; question words found in evidence: 0.333
- categories (PROPOSED): ['evidence_in_unusual_section', 'table_or_figure_context_also_cited']
- **gold evidence:** There is a high real time requirement in real world application, especially in online voice search system. Shenma voice search is one of the most popular mobile search engines in China, and it is a streaming service that intermediate recognition results displayed while users are still speaking. Unid
- top-1 [Experimental setup] 'LSTM RNNs outperform conventional RNNs for speech recognition system, especially deep LSTM RNNs, because of its long-range dependencies more accurately for temporal sequence conditions BIBREF26 , BIBR'
- top-2 [Abstract] 'Recurrent neural networks (RNNs), especially long short-term memory (LSTM) RNNs, are effective network for sequential task like speech recognition. Deeper LSTM models perform well on large vocabulary '
- top-3 [Conclusion] 'We have presented a whole deep unidirectional LSTM parallel training system for LVCSR. The recognition performance improves when the network goes deep. Distillation makes it possible that deep LSTM mo'

## 1706.01678:a6a48de63c1928238b37c2a01c924b852fe752f8
**Q:** Which other methods do they compare with?
- answer types: ['extractive']; evidence paragraphs: 3; gold sections: ['Baselines', 'Procedure to Analyze and Evaluate each step']; question words found in evidence: 1.0
- categories (PROPOSED): ['multi_paragraph_evidence']
- **gold evidence:** For the CNN-Dailymail dataset, the Lead-3 model is considered a strong baseline; both the abstractive BIBREF16 and extractive BIBREF14 state-of-the art methods on this dataset beat this baseline only marginally. The Lead-3 model simply produces the leading three sentences of the document as its summ
- **gold evidence:** For the proxy report section of the AMR bank, we consider the Lead-1-AMR model as the baseline. For this dataset we already have the gold-standard AMR graphs of the sentences. Therefore, we only need to nullify the error introduced by the generator.
- **gold evidence:** In order to compare our summary graph extraction step with the previous work ( BIBREF0 ), we generate the final summary using the same generation method as used by them. Their method uses a simple module based on alignments for generating summary after step-2. The alignments simply map the words in 
- top-1 [Procedure to Analyze and Evaluate each step] 'Step-3: Generation For evaluating the quality of the sentences generated by our method, we compare the summaries generated by the first-1 model and Lead-1-AMR model on the gold-standard dataset. Howev'
- top-2 [Introduction] 'Summarization of large texts is still an open problem in language processing. People nowadays have lesser time and patience to go through large pieces of text which make automatic summarization import'
- top-3 [Effectiveness of the Generator] "In this subsection we evaluate the effectiveness of the sentence generation step. For fair comparison at the generation step we use the gold-standard AMRs and don't perform any extraction in step-2 in"

## 1708.01464:0752d71a0a1f73b3482a888313622ce9e9870d6e
**Q:** what was the baseline?
- answer types: ['extractive']; evidence paragraphs: 1; gold sections: ['Baseline']; question words found in evidence: 0.0
- categories (PROPOSED): ['vocabulary_mismatch']
- **gold evidence:** Results on LangID and NoLangID are compared to the system presented by deri2016grapheme, which is identified in our results as wFST. Their results can be divided into two parts:
- top-1 [Adapted Results] 'On the 229 languages for which deri2016grapheme presented their final results, the LangID version of our system outperforms the baseline by a wide margin. The best performance came with the version of'
- top-2 [Experiments] 'We present experiments with two versions of our sequence-to-sequence model. LangID prepends each training, validation, and test sample with an artificial token identifying the language of the sample. '
- top-3 [Evaluation] 'We use the following three evaluation metrics:\n\nPhoneme Error Rate (PER) is the Levenshtein distance between the predicted phoneme sequences and the gold standard phoneme sequences, divided by the len'

## 1802.06024:585626d18a20d304ae7df228c2128da542d248ff
**Q:** What metrics are used to establish that this makes chatbots more knowledgeable and better at learning and conversation?
- answer types: ['extractive']; evidence paragraphs: 1; gold sections: ['Experiments']; question words found in evidence: 0.111
- categories (PROPOSED): ['vocabulary_mismatch']
- **gold evidence:** Evaluation Metrics. To evaluate the strategy formulation ability, we introduce a measure called Coverage( INLINEFORM0 ), defined as the fraction of total query data instances, for which LiLi has successfully formulated strategies that lead to winning. If LiLi wins on all episodes for a given dataset
- top-1 [Introduction] 'or implicitly learn new knowledge in the conversation process. This seriously limits the scope of their applications. In contrast, we humans constantly learn new knowledge in our conversations. Even i'
- top-2 [Abstract] 'Although chatbots have been very popular in recent years, they still have some serious weaknesses which limit the scope of their applications. One major weakness is that they cannot learn new knowledg'
- top-3 [Introduction] 'Chatbots such as dialog and question-answering systems have a long history in AI and natural language processing. Early such systems were mostly built using markup languages such as AIML, handcrafted '

## 1802.08636:f03112b868b658c954db62fc64430bebbaa7d9e0
**Q:** Do they use other evaluation metrics besides ROUGE?
- answer types: ['yes_no']; evidence paragraphs: 1; gold sections: ['Introduction']; question words found in evidence: 0.4
- categories (PROPOSED): ['none_of_the_proposed']
- **gold evidence:** We report results on the CNN and DailyMail news highlights datasets BIBREF14 which have been recently used as testbeds for the evaluation of neural summarization systems. Experimental results show that when evaluated automatically (in terms of ROUGE), our model outperforms state-of-the-art extractiv
- top-1 [Related Work] 'Recent deep learning methods BIBREF63 , BIBREF64 , BIBREF9 , BIBREF10 learn continuous features without any linguistic preprocessing (e.g., named entities). Like traditional methods, these approaches '
- top-2 [The Pitfalls of Cross-Entropy Loss] 'These discrepancies render the model less efficient at ranking sentences for the summarization task. Instead of maximizing the likelihood of the ground-truth labels, we could train the model to predic'
- top-3 [The Pitfalls of Cross-Entropy Loss] 'Table TABREF6 (last column) shows candidate summaries ranked according to the mean of ROUGE-1, ROUGE-2, and ROUGE-L F INLINEFORM0 scores. Interestingly, multiple top ranked summaries have reasonably h'

## 1809.01202:a4d115220438c0ded06a91ad62337061389a6747
**Q:** What types of social media did they consider?
- answer types: ['extractive']; evidence paragraphs: 1; gold sections: ['Dataset']; question words found in evidence: 0.0
- categories (PROPOSED): ['vocabulary_mismatch']
- **gold evidence:** We created our own causal explanation dataset by collecting 3,268 random Facebook status update messages. Three well-trained annotators manually labeled whether or not each message contains the causal explanation and obtained 1,598 causality messages with substantial agreement ( $\kappa =0.61$ ). We
- top-1 [Introduction] 'Prevailing approaches for social media analyses, utilizing traditional linear models or bag of words models (e.g., SVM trained with n-gram, part-of-speech (POS) tags, or lexicon-based features) alone '
- top-2 [Complete Pipeline] 'We had an inevitable limitation on the size of our dataset, since there is no other causality dataset over social media and the annotation required an intensive iterative process. This might have limi'
- top-3 [Experiment] 'We explored three types of models (RBF SVM, Linear SVM, and Random Forest Classifier) which have previously been shown empirically useful for the language analysis in social media. We filtered out low'

## 1809.06537:41b70699514703820435b00efbc3aac4dd67560a
**Q:** what civil field is the dataset about?
- answer types: ['extractive']; evidence paragraphs: 1; gold sections: ['Experiments']; question words found in evidence: 0.667
- categories (PROPOSED): ['none_of_the_proposed']
- **gold evidence:** To evaluate the proposed LRC framework and the AutoJudge model, we carry out a series of experiments on the divorce proceedings, a typical yet complex field of civil cases. Divorce proceedings often come with several kinds of pleas, e.g. seeking divorce, custody of children, compensation, and mainte
- top-1 [Abstract] 'interactions among facts, pleas, and laws. In experiments, we construct a real-world civil case dataset for LRC. Experimental results on this dataset demonstrate that our model achieves significant im'
- top-2 [Conclusion] 'In the future, we can explore the following directions: (1) Limited by the datasets, we can only verify our proposed model on divorce proceedings. A more general and larger dataset will benefit the re'
- top-3 [Introduction] 'of the judicial system. Besides, automated judgment system can help in improving juridical equality and transparency. From another perspective, there are currently 7 times much more civil cases than c'

## 1810.12091:9b4dc790e4ff49562992aae4fad3a38621fadd8b
**Q:** what are the existing approaches?
- answer types: ['extractive']; evidence paragraphs: 1; gold sections: ['Variants and Baseline Methods']; question words found in evidence: 0.0
- categories (PROPOSED): ['vocabulary_mismatch']
- **gold evidence:** We will refer to our model as EGEL (Embedding GEographic Locations), and will consider the following variants. EGEL-Tags only uses the information from the Flickr tags (i.e. component INLINEFORM0 ), without using any negative examples and without feature selection. EGEL-Tags+NS is similar to EGEL-Ta
- top-1 [Embeddings for geographic information] 'Several methods also use word embedding models to learn representations of Points-of-Interest (POIs) that can be used for predicting user visits BIBREF33 , BIBREF34 , BIBREF35 . These works use the ma'
- top-2 [Introduction] 'Users of photo-sharing websites such as Flickr often provide short textual descriptions in the form of tags to help others find the images. With the availability of GPS systems in current electronic d'
- top-3 [Tag Based Location Embedding] 'where: INLINEFORM0\n\nwith INLINEFORM0 the set of all tags, and INLINEFORM1 the set of locations.\n\nTag selection. Inspired by BIBREF50 , we use a term selection method in order to focus on the tags that'

## 1902.09666:682e26262abba473412f68cbeb5f69aa3b9968d7
**Q:** What are the differences between this dataset and pre-existing ones?
- answer types: ['extractive']; evidence paragraphs: 1; gold sections: ['Introduction']; question words found in evidence: 0.0
- categories (PROPOSED): ['vocabulary_mismatch']
- **gold evidence:** Recently, Waseem et. al. ( BIBREF12 ) acknowledged the similarities among prior work and discussed the need for a typology that differentiates between whether the (abusive) language is directed towards a specific individual or entity or towards a generalized group and whether the abusive content is 
- top-1 [Hierarchically Modelling Offensive Content] 'In the OLID dataset, we use a hierarchical annotation model split into three levels to distinguish between whether language is offensive or not (A), and type (B) and target (C) of the offensive langua'
- top-2 [Offensive Language Target Identification] 'The results of the offensive target identification experiment are reported in Table TABREF20 . Here the systems were trained to distinguish between three targets: a group (GRP), an individual (IND), o'
- top-3 [Conclusion and Future Work] 'In future work, we would like to make a cross-corpus comparison of OLID and datasets annotated for similar tasks such as aggression identification BIBREF2 and hate speech detection BIBREF8 . This comp'

## 1903.03530:2c85865a65acd429508f50b5e4db9674813d67f2
**Q:** Which data do they use as a starting point for the dialogue dataset?
- answer types: ['abstractive', 'extractive']; evidence paragraphs: 2; gold sections: ['Data Preparation', 'Linguistic Characterization on Seed Data']; question words found in evidence: 0.5
- categories (PROPOSED): ['multi_paragraph_evidence']
- **gold evidence:** We used recordings of nurse-initiated telephone conversations for congestive heart failure patients undergoing telemonitoring, post-discharge from the hospital. The clinical data was acquired by the Health Management Unit at Changi General Hospital. This research study was approved by the SingHealth
- **gold evidence:** To analyze the linguistic structure of the inquiry-response pairs in the entire 41-hour dataset, we randomly sampled a seed dataset consisting of 1,200 turns and manually categorized them to different types, which are summarized in Table TABREF14 along with the corresponding occurrence frequency sta
- top-1 [Motivation of Approach] 'In healthcare, conversation data is even scarcer due to privacy issues. Crowd-sourcing is an efficient way to annotate large quantities of data, but less suitable for healthcare scenarios, where domai'
- top-2 [Reading Comprehension] 'While language understanding tasks in dialogue such as domain identification BIBREF15 , slot filling BIBREF16 and user intent detection BIBREF17 have attracted much research interest, work in dialogue'
- top-3 [Abstract] 'Data for human-human spoken dialogues for research and development are currently very limited in quantity, variety, and sources; such data are even scarcer in healthcare. In this work, we investigate '

## 1905.10851:49eb52b3ec0647e165a5e41488088c80a20cc78f
**Q:** What aspects of discussion are relevant to instructor intervention, according to the attention mechanism?
- answer types: ['extractive']; evidence paragraphs: 1; gold sections: ['Discussion']; question words found in evidence: 0.25
- categories (PROPOSED): ['none_of_the_proposed']
- **gold evidence:** In order to understand if context inference is useful to intervention prediction, we ablate the attention components and experiment with the vanilla hierarchical LSTM model. Row 3 of Table TABREF17 shows the macro averaged result from this experiment. The UPA and PPA attention models better the vani
- top-1 [Introduction] 'In this paper, we improve the state-of-the-art for instructor intervention in MOOC forums. We propose the first neural models for this prediction problem. We show that modelling the thread structure a'
- top-2 [Abstract] 'Due to time constraints, course instructors often need to selectively participate in student discussion threads, due to their limited bandwidth and lopsided student--instructor ratio on online forums.'
- top-3 [Contextual Attention Models] 'When they intervene, instructors either pay attention to a specific post or a series of posts, which trigger their reply. However, instructors rarely explicitly indicate to which post(s) their interve'

## 1906.07662:fe2666ace293b4bfac3182db6d0c6f03ea799277
**Q:** Why challenges does word segmentation in Vietnamese pose?
- answer types: ['abstractive', 'extractive']; evidence paragraphs: 1; gold sections: ['CONCLUSIONS AND FUTURE WORKS']; question words found in evidence: 0.6
- categories (PROPOSED): ['none_of_the_proposed']
- **gold evidence:** There are several challenges on supervised learning approaches in future work. The first challenge is to acquire very large Vietnamese corpus and to use them in building a classifier, which could further improve accuracy. In addition, applying linguistics knowledge on word context to extract useful 
- top-1 [Abstract] 'Word segmentation is the first step of any tasks in Vietnamese language processing. This paper reviews stateof-the-art approaches and systems for word segmentation in Vietnamese. To have an overview o'
- top-2 [CONCLUSIONS AND FUTURE WORKS] 'This study reviewed state-of-the-art approaches and systems of Vietnamese word segmentation. The review pointed out common features and methods used in Vietnamese word segmentation studies. This study'
- top-3 [TOOLKITS] 'vnTokenizer and JVnSegmenter are two famous segmentation toolkits for Vietnamese word segmentation. Both two word segmentation toolkits are implemented the word segmentation data process in Figure 2. '

## 1909.09587:55569d0a4586d20c01268a80a7e31a17a18198e2
**Q:** what does the model learn in zero-shot setting?
- answer types: ['extractive']; evidence paragraphs: 2; gold sections: ['Zero-shot Transfer with Multi-BERT ::: Experimental Setup and Data']; question words found in evidence: 0.2
- categories (PROPOSED): ['vocabulary_mismatch', 'multi_paragraph_evidence']
- **gold evidence:** We have training and testing sets in three different languages: English, Chinese and Korean. The English dataset is SQuAD BIBREF2. The Chinese dataset is DRCD BIBREF14, a Chinese RC dataset with 30,000+ examples in the training set and 10,000+ examples in the development set. The Korean dataset is K
- **gold evidence:** The pre-trained multi-BERT is the official released one. This multi-lingual version of BERT were pre-trained on corpus in 104 languages. Data in different languages were simply mixed in batches while pre-training, without additional effort to align between languages. When fine-tuning, we simply adop
- top-1 [Zero-shot Transfer with Multi-BERT] 'Multi-BERT has showcased its ability to enable cross-lingual zero-shot learning on the natural language understanding tasks including XNLI BIBREF19, NER, POS, Dependency Parsing, and so on. We now see'
- top-2 [Abstract] 'Because it is not feasible to collect training data for every language, there is a growing interest in cross-lingual transfer learning. In this paper, we systematically explore zero-shot cross-lingual'
- top-3 [What Does Zero-shot Transfer Model Learn? ::: Unseen Language Dataset] 'It has been shown that extractive QA tasks like SQuAD may be tackled by some language independent strategies, for example, matching words in questions and context BIBREF20. Is zero-shot learning feasi'

## 1910.06592:483a699563efcb8804e1861b18809279f21c7610
**Q:** Was the approach used in this work to detect fake news fully supervised?
- answer types: ['yes_no']; evidence paragraphs: 1; gold sections: ['Methodology']; question words found in evidence: 0.125
- categories (PROPOSED): ['vocabulary_mismatch']
- **gold evidence:** Given a news Twitter account, we read its tweets from the account's timeline. Then we sort the tweets by the posting date in ascending way and we split them into $N$ chunks. Each chunk consists of a sorted sequence of tweets labeled by the label of its corresponding account. We extract a set of feat
- top-1 [Introduction] 'than an account level, since some accounts mix fake with real news. However, these annotations are extremely costly and time consuming – i.e., due to high volume of available tweets Consequently, a fi'
- top-2 [Abstract] 'We present an approach to detect fake news in Twitter at the account level using a neural recurrent model and a variety of different semantic and stylistic features. Our method extracts a set of featu'
- top-3 [Introduction] 'The main obstacle for detecting suspicious Twitter accounts is due to the behavior of mixing some real news with the misleading ones. Consequently, we investigate ways to detect suspicious accounts by'

## 1911.07555:e5c8e9e54e77960c8c26e8e238168a603fcdfcc6
**Q:** Does the algorithm improve on the state-of-the-art methods?
- answer types: ['abstractive', 'yes_no']; evidence paragraphs: 1; gold sections: ['Results and Analysis']; question words found in evidence: 0.0
- categories (PROPOSED): ['vocabulary_mismatch', 'table_or_figure_context_also_cited']
- **gold evidence:** The average classification accuracy results are summarised in Table TABREF9. The accuracies reported are for classifying a piece of text by its specific language label. Classifying text only by language group or family is a much easier task as reported in BIBREF8.
- top-1 [Results and Analysis] 'The accuracy of the proposed algorithm seems to be dependent on the support of the lexicon. Without a good lexicon a non-stacked naive Bayesian classifier might even perform better.\n\nThe execution per'
- top-2 [Related Works] 'The focus of this section is on recently published datasets and LID research applicable to the South African context. An in depth survey of algorithms, features, datasets, shared tasks and evaluation '
- top-3 [Conclusion] 'LID of short texts, informal styles and similar languages remains a difficult problem which is actively being researched. The proposed algorithm was evaluated on three existing datasets and compared t'

## 1912.06262:9a8b9ea3176d30da2453cac6e9347737c729a538
**Q:** what were their performance results?
- answer types: ['extractive']; evidence paragraphs: 1; gold sections: ['Experiments ::: Results']; question words found in evidence: 0.0
- categories (PROPOSED): ['vocabulary_mismatch']
- **gold evidence:** With the above hyperparameter setting, the hybrid NER model achieved a F1 score of $0.995$ on synthesized queries and $0.948$ on clinical notes while the i2b2 NER model achieved a F1 score of $0.441$ on synthesized queries and $0.927$ on clinical notes (See Table TABREF23).
- top-1 [Experiments ::: Results] 'Since there was no ground truth available for the retrieved terms, we randomly picked a few samples to assess its performance. Some example outputs of our complete framework on real user queries are s'
- top-2 [Experiments ::: Hyperparameter Tuning] 'We defined the following hyperparameter search space:\n\nembeddings: [“ELMo on pubmed”, “stacked flair on pubmed”],\n\nhidden_size: [128, 256],\n\nlearning_rate: [0.05, 0.1],\n\nmini_batch_size: [32, 64, 128]'
- top-3 [Experiments ::: Setup] 'To show the effects of using the hybrid training data, we trained two models of the same architecture and hyperparameters. One model was trained on the hybrid data and will be referred to as hybrid NE'

## 1912.08960:63a1cbe66fd58ff0ead895a8bac1198c38c008aa
**Q:** Which existing models are evaluated?
- answer types: ['extractive']; evidence paragraphs: 1; gold sections: ['Experimental Setup ::: Models']; question words found in evidence: 0.333
- categories (PROPOSED): ['none_of_the_proposed']
- **gold evidence:** We experiment with two image captioning models: the Show&Tell model BIBREF0 and the LRCN1u model BIBREF1. Both models follow the basic encoder-decoder architecture design that uses a CNN encoder to condense the visual information into an image embedding, which in turn conditions an LSTM decoder to g
- top-1 [Discussions and Conclusions] 'Evaluation metrics are required as a proxy for performance in real applications. As such, they should, as far as possible, allow measurement of fundamental aspects of the performance of models on task'
- top-2 [Introduction] 'Practical evaluation of GTD is currently only possible on synthetic data. We construct a range of datasets designed for image captioning evaluation. We call this diagnostic evaluation benchmark ShapeW'
- top-3 [Related work ::: Synthetic datasets] 'In this work, we develop the evaluation datasets within the ShapeWorld framework. ShapeWorld is a controlled data generation framework consisting of abstract colored shapes (see Figure FIGREF1 for an '

## 2001.01269:81d193672090295e687bc4f4ac1b7a9c76ea35df
**Q:** What baseline method is used?
- answer types: ['abstractive', 'extractive']; evidence paragraphs: 1; gold sections: ['Related Work']; question words found in evidence: 0.333
- categories (PROPOSED): ['none_of_the_proposed']
- **gold evidence:** In a recent work on sentiment analysis in Turkish BIBREF10, they learn embeddings using Turkish social media. They use the word2vec algorithm, create several unsupervised hand-crafted features, generate document vectors and feed them as input into the support vector machines (SVM) approach. We outpe
- top-1 [Methodology ::: Combination of the Word Embeddings] 'In addition to using the three approaches independently, we also combined all the matrices generated in the previous approaches. That is, we concatenate the reduced forms (SVD - U) of corpus-based, di'
- top-2 [Experiments ::: Results] 'We obtained varying accuracies as shown in Table TABREF17. “3 feats" features are those hand-crafted features we extracted, which are the minimum, mean, and maximum polarity scores of the reviews as e'
- top-3 [Experiments ::: Results] 'Since the TDK dictionary covers most of the domain-specific vocabulary used in the movie reviews, the dictionary method performs well. However, the dictionary lacks many of the words, occurring in the'
