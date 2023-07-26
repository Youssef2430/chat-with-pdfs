css = '''
<style>
.chat-message {
    padding: 0.5rem; border-radius: 0.5rem; display: flex; float: right; margin-top: 1rem;
    width: 80%;
}
.chat-message-user {
    padding: 0.5rem; border-radius: 0.5rem; display: flex; float: left; margin-top: 1rem; background-color: #EE7A07;
    width: 80%;
}

.chat-message.bot {
    background-color: #313131
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .message {
  padding: 0 1.5rem;
  color: #fff;
}

.chat-message-user .message {
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message-user">   
    <div class="message">{{MSG}}</div>
</div>
'''