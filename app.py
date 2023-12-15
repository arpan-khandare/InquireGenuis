from mcq import getMCQs
import streamlit as st
import random


st.title('Sentence2MCQ')

sentence = st.text_input("Sentence")
submit = st.button("Submit")
if st.session_state.get('submit') != True:
    st.session_state['submit'] = submit
if st.session_state['submit']:
    question,answer,distractors,meaning = getMCQs(sentence)

    options = [answer] + random.sample(distractors, 3)
    random.shuffle(options)

    selected_option = st.radio(question, options)

    if (st.button("Submit option")):
        st.session_state['submit'] = False
        if selected_option == answer:
            st.balloons()
            st.write("Congratulations! You selected the correct option.")
        else:
            st.write("Sorry, that is not the correct option.")
            st.markdown(f"Correct Answer is **:green[{answer}]**")


    