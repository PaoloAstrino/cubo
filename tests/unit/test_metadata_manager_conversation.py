from cubo.storage.metadata_manager import get_metadata_manager


def test_clear_conversation_removes_messages_but_keeps_conversation():
    mm = get_metadata_manager()
    conv_id = mm.create_conversation(title="ClearTest")
    mm.add_message(conv_id, "user", "hi")
    mm.add_message(conv_id, "assistant", "welcome")

    msgs = mm.get_conversation_messages(conv_id)
    assert len(msgs) == 2

    removed = mm.clear_conversation(conv_id)
    assert removed is True

    msgs_after = mm.get_conversation_messages(conv_id)
    assert msgs_after == []

    conv = mm.get_conversation(conv_id)
    assert conv is not None
