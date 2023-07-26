css = '''
<style>
.chat-message {
    padding: 0.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #EE7A07
}
.chat-message.bot {
    background-color: #313131
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 60px;
  max-height: 60px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
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
<div class="chat-message user">   
    <div class="message">{{MSG}}</div>
</div>
'''