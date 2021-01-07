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
        correct_style = "background:#8f8;padding:10px 25px 10px 25px;width:40%;float:right;line-height:16px;border-radius:16px;"
        incorrect_style = "background:#f88;padding:10px 25px 10px 25px;width:40%;float:right;line-height:16px;border-radius:16px;"
        selfcheck_style = "background:#ff8;padding:10px 25px 10px 25px;width:40%;float:right;line-height:16px;border-radius:16px;"
        self.header = """<style>
            .correct { %s !important; }
            .incorrect { %s !important; }
            .selfcheck { %s !important; }
            .correct > div > p { margin-top:8px; } 
            .incorrect > div > p { margin-top:8px; } 
            .selfcheck > div > p { margin-top:8px; } 
            </style>"""%(correct_style,incorrect_style,selfcheck_style)

    def display(self,font_awesome=True):
        """Displays this quiz in the Jupyter notebook."""
        if font_awesome:
            display(HTML('<link rel="stylesheet" href="//stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"/>'))
        display(self.widget())
    
    def widget(self):
        """Returns the widget for this quiz"""
        if len(self.questions) == 0:
            return widgets.HTML("No questions in this quiz")
        return widgets.VBox([widgets.HTML(self.header)] + self.questions + [self.submit])

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
                submission. This may contain HTML / Latex.
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
                    feedback_str = '<p>{}'.format('<p>'.join(feedback_strs))
                else:
                    feedback_str = ''
                label = widgets.HTMLMath('Incorrect.{}<p>Please try again.'.format(feedback_str))
                label.add_class('incorrect')
                remove = container.children[-1]
                container.children = container.children[:-1] + (label,)
                remove.close()

            def on_correct(value):
                feedback_str = ''
                #label = widgets.HTML('<div style"{}">Correct!{}</div>'.format(correct_style,feedback_str))
                label = widgets.HTMLMath('Correct!{}'.format(feedback_str))
                label.add_class('correct')
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
                feedback_str = '' if feedback is None or feedback[value_to_index[value]] is None else '<p>{}'.format(feedback[value_to_index[value]])
                label = widgets.HTMLMath('Incorrect.{}<p>Please try again.'.format(feedback_str))
                label.add_class('incorrect')
                remove = container.children[-1]
                container.children = container.children[:-1] + (label,)
                remove.close()

            def on_correct(value):
                feedback_str = '' if feedback is None or feedback[value_to_index[value]] is None else '<p>{}'.format(feedback[value_to_index[value]])
                label = widgets.HTMLMath('Correct!{}'.format(feedback_str))
                label.add_class('correct')
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
        
    def add_short_answer(self,prompt,answers,
                            number=None,
                            feedback=None,
                            case_sensitive=False,
                            edit_distance=0):
        """
        Adds IPython widgets showing a short-entry quiz item, which can accept
        one or more free-text answers.  The answer will be validated if the
        edit distance between the entered text and the 

        Args:
            prompt (str): the prompt. Can contain HTML / Latex.
            answers (list of str): the answers to be compared as feedback.
            number (int or str, optional): The problem number. Incremented from
                1 by default.
            case_sensitive (bool, optional): Whether to consider capitalization
                errors.
            feedback (str, list of str, or dict of str->str, optional): Answer
                feedback, provided upon submission. If a list of str, this must
                have the same length as answers, and will match on the best
                answer.  If a dict of str, the best match to keys here (within
                edit_distance will be displayed.  This latter format allows you
                to give feeback on common incorrect answers.
                
                Feedback may contain HTML, but Latex is not supported yet!
            edit_distance (int, optional): the acceptable edit distance to one
                of the provided answers.  By default have to match exactly.
        """
        self.json_record.append({'type':'short-answer',
                                'prompt':prompt,
                                'answers':answers,
                                'number':number,
                                'feedback':feedback,
                                'case_sensitive':case_sensitive,
                                'edit_distance':edit_distance})

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
        
        form = widgets.Text(
            value='',
            placeholder='Enter your text here',
            description=' ',
            layout={'width': '60%'}, # If the items' names are long   
            disabled=False
        )
        label = widgets.Label('')
        container = widgets.Box([form,label])
        widget_items.append(container)

        def edit_dist(a, b):
            return len(b) if not a else len(a) if not b \
                 else min(edit_dist(a[1:], b[1:])+(a[0] != b[0]),
                          edit_dist(a[1:], b)+1,
                          edit_dist(a, b[1:])+1)

        def on_submit(event):
            answer = form.value.strip()
            if not case_sensitive:
                answer = answer.lower()
            accept = False
            lowest_distance = edit_distance+1
            bestMatch = None
            for i,a in enumerate(answers):
                if case_sensitive:
                    d = edit_dist(answer,a)
                else:
                    d = edit_dist(answer,a.lower())
                if d < lowest_distance:
                    accept = True
                    bestMatch = i
                    lowest_distance = d
            feedback_str = None
            if isinstance(feedback,str):
                chosen_feedback = feedback
            if accept:
                if isinstance(feedback,list):
                    feedback_str = feedback[bestMatch]
                elif isinstance(feedback,dict):
                    feedback_str= feedback.get(answers[bestMatch],None)
            else:
                if isinstance(feedback,dict):
                    lowest_distance = edit_distance+1
                    for (k,v) in feedback.items():
                        if case_sensitive:
                            d = edit_dist(answer,k)
                        else:
                            d = edit_dist(answer,k.lower())
                        if d < lowest_distance:
                            feedback_str= v
                            lowest_distance = d
            if feedback_str is None:
                feedback_str = ''
            else:
                feedback_str = '<p>'+feedback_str
            if accept:
                label = widgets.HTMLMath('Correct!{}'.format(feedback_str))
                label.add_class('correct')
                remove = container.children[-1]
                container.children = container.children[:-1] + (label,)
                remove.close()
            else:
                label = widgets.HTMLMath('Incorrect.{}<p>Please try again.'.format(feedback_str))
                label.add_class('incorrect')
                remove = container.children[-1]
                container.children = container.children[:-1] + (label,)
                remove.close()

        def clear_label(change):
            remove = container.children[-1]
            container.children = container.children[:-1] + (widgets.Label(''),)
            remove.close()                

        form.observe(clear_label)

        self.submit.on_click(on_submit)
        self.questions.append(widgets.VBox(widget_items))

    def add_freeform(self,prompt,answer,
                            number=None,
                            lines=None):
        """
        Adds IPython widgets showing a freeform quiz item, which simply 
        displays the answer for the student to compare.

        Args:
            prompt (str): the prompt. Can contain HTML / Latex.
            answer (str): the answer to be provided as feedback.
            number (int or str, optional): The problem number. Incremented from
                1 by default.
            lines (int, optional): the number of lines in the student's entry
                form.  By default estimated from the length of the answer.
        """
        self.json_record.append({'type':'freeform',
                                'prompt':prompt,
                                'answer':answer,
                                'number':number,
                                'lines':lines})

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
        
        if lines is None:
            lines = max(len(answer)//60,1)
        
        widget_items = []
        prompt = self.number_format.format(number) + prompt
        prompt_widget = widgets.HTMLMath(prompt)
        widget_items.append(prompt_widget)

        form = widgets.Textarea(
            value='',
            placeholder='Enter your text here',
            description=' ',
            layout={'width': '60%'}, # If the items' names are long   
            rows=lines,
            disabled=False
        )
        label = widgets.Label('')
        container = widgets.Box([form,label])
        widget_items.append(container)

        def on_submit(event):
            label = widgets.HTMLMath(answer)
            label.add_class('selfcheck')
            remove = container.children[-1]
            container.children = container.children[:-1] + (label,)
            remove.close()

        def clear_label(change):
            remove = container.children[-1]
            container.children = container.children[:-1] + (widgets.Label(''),)
            remove.close()                

        form.observe(clear_label)

        self.submit.on_click(on_submit)
        self.questions.append(widgets.VBox(widget_items))

    def add_from_json(self,json_obj):
        if not isinstance(json_obj,dict):
            raise ValueError("json_obj must be a dict")
        type = json_obj['type']
        args = json_obj.copy()
        del args['type']
        if type == 'multiple-choice':
            self.add_multiple_choice(**args)
        elif type == 'short-answer':
            self.add_short_answer(**args)
        elif type == 'freeform':
            self.add_freeform(**args)
        else:
            raise ValueError("Invalid 'type': "+type)

BASE_PATH = os.path.split(__file__)[0]

def show(fn):
    import json
    q = Quiz()
    with open(os.path.join(BASE_PATH,fn+'.json'),'r') as f:
        jsonobj = json.load(f)
    q.from_json(jsonobj)
    q.display()


def self_test():
    display(widgets.HTML("<h2>Here's a test quiz</h2>"))
    q = Quiz()
    q.add_multiple_choice('Whats your favorite pizza topping?',['Pepperoni','Mushrooms','Pineapple'],0)
    q.add_multiple_choice('Whats My name?',['Joe','Kris','Alan','Richard'],1,feedback=['<i>Not him</i>','Yep.\nHis last name is Hauser.','Right family...',None])
    q.add_multiple_choice('Whats My name 2?',['<b>Joe</b>','Kriskjhasdfasldfkjasldfkaslkdfj','Alan','Richard'],1,feedback=['<i>Not him</i>','Yep.\nHis last name is Hauser.','Right family...',None])
    q.add_short_answer('A short answer question: fill in the blank.  Mary had a little _____.',['lamb'],feedback={'lamb':'Great!','sheep':'Younger than that'},edit_distance=1)
    q.add_freeform('A freeform question...',"Here's the answer, $2\pi r$, you'll have to check whether it makes sense yourself")
    q.display()
