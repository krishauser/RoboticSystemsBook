from IPython.display import display,HTML
import ipywidgets as widgets
import random
import os

class Quiz:
    """Creates an IPython multiple choice quiz."""
    def __init__(self):
        self.questions = []
        self.submit = None
        self.number = 1
        self.number_format = "Q{}. "
        self.answer_format = "{}. "
        self.json_record = []

    def to_json(self):
        """Converts quiz to a JSON object."""
        return self.json_record

    def from_json(self,json_obj):
        """Converts quiz from a JSON object."""
        if not isinstance(json_obj,(list,tuple)):
            raise ValueError("json_obj must be a set of questions")
        for item in json_obj:
            self.add_from_json(item)

    def add_multiple_choice(self,prompt,options,answer,
                            number=None,
                            displayed=None,
                            feedback=None,
                            randomize=True):
        """
        Adds IPython widgets showing a multiple-choice quiz item.

        If the items contain HTML or Latex, this will automatically place the
        items in a grid above the answer form.

        Args:
            prompt (str): the prompt. Can contain HTML / Latex.
            options (list of str): the answer items. If the items have HTML or
                Latex, then they are drawn in a separate layout since Jupyter
                doesn't support fancy options.
            answer (int or list): the index of the answer for single-answer
                questions, or a list of answers if more than one can be
                selected.
            number (int or str, optional): The problem number.  Incremented from
                1 by default.
            displayed (int, optional): If provided, then a subset of options
                are presented.
            feedback (list of str, optional): Answer feedback, provided upon
                submission. This may contain HTML, but Latex is not supported
                yet!
            randomize (bool, optional): whether to randomize the answers.
        """
        if isinstance(answer,(list,tuple)):
            for a in answer:
                assert a >= 0 and a < len(options)
        else:
            assert answer >= 0 and answer < len(options)
        if feedback is not None:
            assert len(feedback) == len(options)
        self.json_record.append({'type':'multiple-choice',
                                'prompt':prompt,
                                'options':options,
                                'answer':answer,
                                'number':number,
                                'displayed':displayed,
                                'feedback':feedback,
                                'randomize':randomize})

        if self.submit is None:
            self.submit = widgets.Button(
                description='Submit',
                disabled=False,
                button_style='', # 'success', 'info', 'warning', 'danger' or ''
                tooltip='Submit your answers for grading',
                icon='check' # (FontAwesome names without the `fa-` prefix)
              )

        if number is None:
            number = str(self.number)
            self.number += 1
        if isinstance(number,int):
            self.number = number
        
        widget_items = []
        prompt = self.number_format.format(number) + prompt
        prompt_widget = widgets.HTMLMath(prompt)
        widget_items.append(prompt_widget)

        if isinstance(answer,(list,tuple)):
            widget_items.append(widgets.HTML("<i>Select all that apply</i>"))

        prefixes = [chr(ord('A') + i)  for i in range(len(options))]
        fancy = False
        opts_long = False
        for opt in options:
            if '$' in opt or ('<' in opt and '>' in opt):
                fancy = True
                break
            if len(opt) > 80:
                opts_long = True
        if opts_long:
            fancy = True
            opts_long = False  #can't get this working well...
        
        #subsample and scramble if needed!
        order = list(range(len(options)))
        if displayed:
            displayed = min(displayed,len(options))
            answers = answer if isinstance(answer,(list,tuple)) else [answer]
            non_answers = [i for i in range(len(options)) if i not in answers]
            selected_indices = random.sample(range(len(non_answers)), max(displayed - len(answers),len(non_answers)))
            included_options = [False]*len(options)
            for i in answers:
                included_options[i] = True
            for i in selected_indices:
                included_options[i] = True
            order = [i for i in range(len(options)) if included_options[i]]
        if randomize:
            order = random.sample(order, len(order))
        inv_order = dict()
        for i,j in enumerate(order):
            inv_order[j] = i

        options = [options[i] for i in order]
        if isinstance(answer,(list,tuple)):
            answer = [inv_order[a] for a in answer]
        else:
            answer = inv_order[answer]
        if feedback is not None:
            feedback = [feedback[i] for i in order]

        if fancy:
            #display options and entry form separately
            items = [widgets.HTMLMath(opt,layout=widgets.Layout(height='auto', width='auto')) for opt in options]
            labels = [widgets.HTMLMath('<b>'+pref+'</b>',layout=widgets.Layout(height='auto', width='20px')) for pref in prefixes]
            labeled_items = []
            for i,l in zip(items,labels):
                labeled_items.append(widgets.Label())
                labeled_items.append(l)
                labeled_items.append(i)
            option_widget = widgets.GridBox(children=labeled_items,
                              layout=widgets.Layout(
                                  width='auto',
                                  grid_template_rows='auto '*len(items),
                                  grid_template_columns='3% 2% 94%'))
            widget_items.append(option_widget)
            opts = prefixes
        else:
            #display inline
            opts = [self.answer_format.format(pref) + opt for pref,opt in zip(prefixes,options)]
        if opts_long:
            incorrect_style = "background:#f88;padding:10px 25px 10px 25px;width:auto;line-height:16px;"
            correct_style = "background:#8f8;padding:10px 25px 10px 25px;width:auto;line-height:16px;"
        else:
            incorrect_style = "background:#f88;padding:15px 25px 15px 25px;margin-left:20px;height:100%;width:auto;line-height:16px;"
            correct_style = "background:#8f8;padding:15px 25px 15px 25px;margin-left:20px;height:100%;width:auto;line-height:16px;"
        value_to_index = dict((opt,i) for i,opt in enumerate(opts))

        if feedback is not None:
            #convert endlines of plain strings to HTML
            for i,f in enumerate(feedback):
                if f is not None and ('<' not in f or '>' not in f):
                    feedback[i] = f.replace('\n','<p>')

        if isinstance(answer,(list,tuple)):
            value_in_answer = [False]*len(options)
            for a in answer:
                value_in_answer[a] = True

            choices = widgets.VBox([widgets.CheckBox(value=False,
                                        description=opt,
                                        disabled=False,
                                        indent=False
                                      ) for opt in opts])
            label = widgets.Label('')
            if opts_long:
                container = widgets.VBox([choices,label])
            else:
                container = widgets.HBox([choices,label])
            widget_items.append(container)

            def on_incorrect(value):
                feedback_strs = []
                if feedback is not None:
                    for i in len(value):
                        if value[i] != value_in_answer[i]:
                            if feedback[i] is not None:
                                feedback_strs.append(feedback[i])
                if feedback_strs:
                    feedback_str = '<ul>{}</ul>'.format('<ul>'.join(feedback_strs))
                else:
                    feedback_str = ''
                label = widgets.HTML('<div style="{}">Incorrect.{}<p>Please try again.</div>'.format(incorrect_style,feedback_str))
                remove = container.children[-1]
                container.children = container.children[:-1] + (label,)
                remove.close()

            def on_correct(value):
                feedback_str = ''
                label = widgets.HTML('<div style"{}">Correct!{}</div>'.format(correct_style,feedback_str))
                remove = container.children[-1]
                container.children = container.children[:-1] + (label,)
                remove.close()

            def on_submit(event):
                answers = [c.value for c in choices.children ]
                if answers == value_in_answer:
                  on_correct(answers)
                else:
                  on_incorrect(answers)

            def clear_label(change):
                remove = container.children[-1]
                container.children = container.children[:-1] + (widgets.Label(''),)
                remove.close()         

            for c in choices.children:
                c.observe(clear_label)
        else:
            answer_prefix = prefixes[answer]

            choices = widgets.RadioButtons(
                description=" ",
                options=opts,
                #layout={'width': 'max-content'}, # If the items' names are long   
                layout={'width': '60%'}, # If the items' names are long   
                disabled=False
            )
            label = widgets.Label('')
            container = widgets.Box([choices,label])
            #if opts_long:
            #    container = widgets.VBox([choices,label])
            #else:
            #    container = widgets.HBox([choices,label])
            widget_items.append(container)

            def on_incorrect(value):
                feedback_str = '' if feedback is None or feedback[value_to_index[value]] is None else '<ul><li>{}</ul>'.format(feedback[value_to_index[value]])
                label = widgets.HTML('<div style="{}">Incorrect.{}<p>Please try again.</div>'.format(incorrect_style,feedback_str))
                remove = container.children[-1]
                container.children = container.children[:-1] + (label,)
                remove.close()

            def on_correct(value):
                feedback_str = '' if feedback is None or feedback[value_to_index[value]] is None else '<ul><li>{}</ul>'.format(feedback[value_to_index[value]])
                label = widgets.HTML('<div style="{}">Correct!{}</div>'.format(correct_style,feedback_str))
                remove = container.children[-1]
                container.children = container.children[:-1] + (label,)
                remove.close()

            def on_submit(event):
                #print("Choice:",choices.value,"answer",answer_prefix)
                if choices.value.startswith(answer_prefix):
                  on_correct(choices.value)
                else:
                  on_incorrect(choices.value)

            def clear_label(change):
                remove = container.children[-1]
                container.children = container.children[:-1] + (widgets.Label(''),)
                remove.close()                

            choices.observe(clear_label)

        self.submit.on_click(on_submit)
        self.questions.append(widgets.VBox(widget_items))

    def add_from_json(self,json_obj):
        if not isinstance(json_obj,dict):
            raise ValueError("json_obj must be a dict")
        type = json_obj['type']
        prompt = json_obj['prompt']
        answer = json_obj['answer']
        number = json_obj.get('number',None)
        if type == 'multiple-choice':
            options = json_obj['options']
            displayed = json_obj.get('displayed',None)
            feedback = json_obj.get('feedback',None)
            randomize = json_obj.get('randomize',True)
            self.add_multiple_choice(prompt,options,answer,number,displayed,feedback,randomize)

    def display(self,font_awesome=True):
        """Displays this quiz in the Jupyter notebook."""
        if font_awesome:
            display(HTML('<link rel="stylesheet" href="//stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"/>'))
        display(self.widget())
    
    def widget(self):
        """Returns the widget for this quiz"""
        if len(self.questions) == 0:
            return widgets.HTML("No questions in this quiz")
        return widgets.VBox(self.questions + [self.submit])

BASE_PATH = os.path.split(__file__)[0]

def show_forward_kinematics():
    import json
    q = Quiz()
    with open(os.path.join(BASE_PATH,'forward_kinematics.json'),'r') as f:
        jsonobj = json.load(f)
    q.from_json(jsonobj)
    q.display()

def show_inverse_kinematics():
    import json
    q = Quiz()
    with open(os.path.join(BASE_PATH,'inverse_kinematics.json'),'r') as f:
        jsonobj = json.load(f)
    q.from_json(jsonobj)
    q.display()

def show_geometry():
    import json
    q = Quiz()
    with open(os.path.join(BASE_PATH,'geometry.json'),'r') as f:
        jsonobj = json.load(f)
    q.from_json(jsonobj)
    q.display()

def show_motion_planning():
    import json
    q = Quiz()
    with open(os.path.join(BASE_PATH,'motion_planning.json'),'r') as f:
        jsonobj = json.load(f)
    q.from_json(jsonobj)
    q.display()

def self_test():
    display(widgets.HTML("<h2>Here's a test quiz</h2>"))
    q = Quiz()
    q.add_multiple_choice('Whats your favorite pizza topping?',['Pepperoni','Mushrooms','Pineapple'],0)
    q.add_multiple_choice('Whats My name?',['Joe','Kris','Alan','Richard'],1,feedback=['<i>Not him</i>','Yep.\nHis last name is Hauser.','Right family...',None])
    q.add_multiple_choice('Whats My name 2?',['<b>Joe</b>','Kriskjhasdfasldfkjasldfkaslkdfj','Alan','Richard'],1,feedback=['<i>Not him</i>','Yep.\nHis last name is Hauser.','Right family...',None])
    q.display()
