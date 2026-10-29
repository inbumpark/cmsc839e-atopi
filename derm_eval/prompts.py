"""All prompt text in one place. Edit wording here; stage logic lives in stages.py."""

# Proposal section 3.3, verbatim apart from the photo/context phrase, which is adjusted per stage.
BASE_PROMPT = (
    "You are a skilled dermatologist with multiple years of experience observing patients "
    "with a variety of skin conditions. Here {intro} from a patient who has come to you to "
    "determine which skin condition they have. Provide your diagnosis with the name of the "
    "skin condition, your reason for the diagnosis in a few words, then your confidence score "
    "of the diagnosis on a scale of 1 to 5."
)

ANSWER_FORMAT = (
    'Respond only with JSON of the form {"diagnosis": "<name of the skin condition>", '
    '"reason": "<a few words>", "confidence": <integer 1-5>}.'
)

# Stage 3, turn 1: the model chooses which single piece of context to request.
SELECT_PROMPT = (
    "You are a skilled dermatologist with multiple years of experience observing patients "
    "with a variety of skin conditions. Here {intro} from a patient who has come to you to "
    "determine which skin condition they have. Before giving your diagnosis, you may ask the "
    "patient about exactly ONE of the following pieces of information. Choose the one that "
    "would be most useful for your diagnosis.\n{menu}\n"
    'Respond only with JSON of the form {{"requested_field": "<one of the keys above>", '
    '"why": "<a few words>"}}.'
)

# Cascade mode: follow-ups appended to the stage-1 conversation.
CASCADE_SELECT_PROMPT = (
    "Before finalizing, you may ask the patient about exactly ONE of the following pieces of "
    "information. Choose the one that would be most useful for your diagnosis.\n{menu}\n"
    'Respond only with JSON of the form {{"requested_field": "<one of the keys above>", '
    '"why": "<a few words>"}}.'
)
CASCADE_UPDATE_PROMPT = (
    "Additional information from the patient:\n{context}\n"
    "Taking this into account, provide your diagnosis again with the name of the skin "
    "condition, your reason for the diagnosis in a few words, then your confidence score of "
    "the diagnosis on a scale of 1 to 5."
)

# Stage 3 (independent), turn 2: the patient's answer to the requested field.
ANSWER_TO_REQUEST_PROMPT = (
    "The patient answers:\n{context}\n"
    "Now provide your diagnosis with the name of the skin condition, your reason for the "
    "diagnosis in a few words, then your confidence score of the diagnosis on a scale of 1 to 5."
)

DIAGNOSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string"},
        "reason": {"type": "string"},
        "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": ["diagnosis", "reason", "confidence"],
}


def select_schema(field_keys):
    return {
        "type": "object",
        "properties": {
            "requested_field": {"type": "string", "enum": list(field_keys)},
            "why": {"type": "string"},
        },
        "required": ["requested_field", "why"],
    }


def intro(n_images, has_context):
    photo = "is a photo" if n_images == 1 else f"are {n_images} photos"
    return f"{photo} and some context" if has_context else photo


def render_context(context, keys, field_names):
    return "\n".join(f"- {field_names[k]}: {context[k]}" for k in keys)


def render_menu(keys, field_names):
    return "\n".join(f"- {k}: {field_names[k]}" for k in keys)


def diagnosis_prompt(n_images, context, keys, field_names):
    """Stages 1, 2 and 4 (independent mode), and stage 1 of cascade mode."""
    text = BASE_PROMPT.format(intro=intro(n_images, bool(keys)))
    if keys:
        text += "\n\nPatient context:\n" + render_context(context, keys, field_names)
    return text + "\n\n" + ANSWER_FORMAT


def select_prompt(n_images, keys, field_names):
    return SELECT_PROMPT.format(intro=intro(n_images, False), menu=render_menu(keys, field_names))


def answer_to_request_prompt(context, key, field_names):
    return ANSWER_TO_REQUEST_PROMPT.format(context=render_context(context, [key], field_names)) \
        + "\n\n" + ANSWER_FORMAT


def cascade_select_prompt(keys, field_names):
    return CASCADE_SELECT_PROMPT.format(menu=render_menu(keys, field_names))


def cascade_update_prompt(context, keys, field_names):
    return CASCADE_UPDATE_PROMPT.format(context=render_context(context, keys, field_names)) \
        + "\n\n" + ANSWER_FORMAT
