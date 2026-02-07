"""Tests for the Message Definition Parser module."""

import pytest
from pathlib import Path
from textwrap import dedent

from robomind.ros2.message_parser import (
    MessageField,
    MessageConstant,
    MessageDefinition,
    MessageDatabase,
    parse_message_file,
    load_message_database,
    BUILTIN_TYPES,
)


class TestMessageField:
    """Tests for MessageField dataclass."""

    def test_basic_field(self):
        f = MessageField(name="data", field_type="float32")
        assert f.name == "data"
        assert f.field_type == "float32"
        assert f.is_array is False
        assert f.is_builtin is False

    def test_array_field(self):
        f = MessageField(name="ranges", field_type="float32[]", is_array=True)
        assert f.is_array is True

    def test_bounded_array(self):
        f = MessageField(name="data", field_type="uint8[10]", is_array=True, array_size=10)
        assert f.array_size == 10

    def test_builtin_detection(self):
        f = MessageField(name="x", field_type="float64", is_builtin=True)
        assert f.is_builtin is True


class TestMessageConstant:
    """Tests for MessageConstant dataclass."""

    def test_constant(self):
        c = MessageConstant(name="SOLID", constant_type="uint8", value="0")
        assert c.name == "SOLID"
        assert c.constant_type == "uint8"
        assert c.value == "0"


class TestMessageDefinition:
    """Tests for MessageDefinition dataclass."""

    def test_basic_msg(self):
        d = MessageDefinition(
            name="Twist",
            package="geometry_msgs",
            full_name="geometry_msgs/msg/Twist",
            kind="msg",
            fields=[
                MessageField(name="linear", field_type="Vector3"),
                MessageField(name="angular", field_type="Vector3"),
            ],
        )
        assert d.name == "Twist"
        assert d.kind == "msg"
        assert len(d.fields) == 2

    def test_to_dict(self):
        d = MessageDefinition(
            name="Test",
            package="test_pkg",
            full_name="test_pkg/msg/Test",
            kind="msg",
            fields=[MessageField(name="x", field_type="float64", is_builtin=True)],
        )
        result = d.to_dict()
        assert result["name"] == "Test"
        assert result["package"] == "test_pkg"
        assert result["full_name"] == "test_pkg/msg/Test"
        assert len(result["fields"]) == 1

    def test_srv_definition(self):
        d = MessageDefinition(
            name="SetBool",
            package="std_srvs",
            full_name="std_srvs/srv/SetBool",
            kind="srv",
            fields=[],
            request_fields=[MessageField(name="data", field_type="bool", is_builtin=True)],
            response_fields=[
                MessageField(name="success", field_type="bool", is_builtin=True),
                MessageField(name="message", field_type="string", is_builtin=True),
            ],
        )
        assert d.kind == "srv"
        assert len(d.request_fields) == 1
        assert len(d.response_fields) == 2


class TestParseMessageFile:
    """Tests for parse_message_file function."""

    def test_parse_simple_msg(self, tmp_path):
        msg_file = tmp_path / "test_pkg" / "msg" / "TestMsg.msg"
        msg_file.parent.mkdir(parents=True)
        msg_file.write_text(dedent("""\
            # A test message
            float64 x
            float64 y
            float64 z
            string label
        """))

        result = parse_message_file(msg_file, "test_pkg", "msg")
        assert result is not None
        assert result.name == "TestMsg"
        assert result.kind == "msg"
        assert len(result.fields) == 4
        assert result.fields[0].name == "x"
        assert result.fields[0].field_type == "float64"
        assert result.fields[3].name == "label"

    def test_parse_array_fields(self, tmp_path):
        msg_file = tmp_path / "test_pkg" / "msg" / "Arrays.msg"
        msg_file.parent.mkdir(parents=True)
        msg_file.write_text(dedent("""\
            float32[] ranges
            uint8[10] fixed_data
            string[] names
        """))

        result = parse_message_file(msg_file, "test_pkg", "msg")
        assert len(result.fields) == 3
        assert result.fields[0].is_array is True
        assert result.fields[1].is_array is True
        assert result.fields[1].array_size == 10

    def test_parse_constants(self, tmp_path):
        msg_file = tmp_path / "test_pkg" / "msg" / "WithConst.msg"
        msg_file.parent.mkdir(parents=True)
        msg_file.write_text(dedent("""\
            uint8 SOLID=0
            uint8 BLINKING=1
            uint8 mode
        """))

        result = parse_message_file(msg_file, "test_pkg", "msg")
        assert len(result.constants) == 2
        assert result.constants[0].name == "SOLID"
        assert result.constants[0].value == "0"
        assert len(result.fields) == 1

    def test_parse_srv_file(self, tmp_path):
        srv_file = tmp_path / "test_pkg" / "srv" / "TestSrv.srv"
        srv_file.parent.mkdir(parents=True)
        srv_file.write_text(dedent("""\
            string command
            int32 value
            ---
            bool success
            string message
        """))

        result = parse_message_file(srv_file, "test_pkg", "srv")
        assert result.kind == "srv"
        assert len(result.request_fields) == 2
        assert len(result.response_fields) == 2
        assert result.request_fields[0].name == "command"
        assert result.response_fields[0].name == "success"

    def test_parse_action_file(self, tmp_path):
        action_file = tmp_path / "test_pkg" / "action" / "TestAction.action"
        action_file.parent.mkdir(parents=True)
        action_file.write_text(dedent("""\
            int32 target
            ---
            bool success
            ---
            float64 progress
        """))

        result = parse_message_file(action_file, "test_pkg", "action")
        assert result.kind == "action"
        assert len(result.goal_fields) == 1
        assert len(result.result_fields) == 1
        assert len(result.feedback_fields) == 1

    def test_parse_with_comments(self, tmp_path):
        msg_file = tmp_path / "test_pkg" / "msg" / "Commented.msg"
        msg_file.parent.mkdir(parents=True)
        msg_file.write_text(dedent("""\
            # Header comment
            float64 x  # x coordinate
            float64 y  # y coordinate
        """))

        result = parse_message_file(msg_file, "test_pkg", "msg")
        assert len(result.fields) == 2

    def test_parse_complex_types(self, tmp_path):
        msg_file = tmp_path / "test_pkg" / "msg" / "Complex.msg"
        msg_file.parent.mkdir(parents=True)
        msg_file.write_text(dedent("""\
            std_msgs/Header header
            geometry_msgs/Point position
            float64 confidence
        """))

        result = parse_message_file(msg_file, "test_pkg", "msg")
        assert len(result.fields) == 3
        assert result.fields[0].field_type == "std_msgs/Header"
        assert result.fields[0].is_builtin is False

    def test_empty_file(self, tmp_path):
        msg_file = tmp_path / "test_pkg" / "msg" / "Empty.msg"
        msg_file.parent.mkdir(parents=True)
        msg_file.write_text("")

        result = parse_message_file(msg_file, "test_pkg", "msg")
        assert result is not None
        assert len(result.fields) == 0


class TestMessageDatabase:
    """Tests for MessageDatabase class."""

    @pytest.fixture
    def db_with_messages(self, tmp_path):
        """Create a database with test messages."""
        # Create test message files
        msg_dir = tmp_path / "test_pkg" / "msg"
        msg_dir.mkdir(parents=True)

        (msg_dir / "Position.msg").write_text(dedent("""\
            float64 x
            float64 y
            float64 z
        """))

        (msg_dir / "Velocity.msg").write_text(dedent("""\
            float64 linear
            float64 angular
        """))

        srv_dir = tmp_path / "test_pkg" / "srv"
        srv_dir.mkdir(parents=True)
        (srv_dir / "GetState.srv").write_text(dedent("""\
            string query
            ---
            string state
            bool valid
        """))

        db = MessageDatabase()
        db.load_project_messages(tmp_path)
        return db

    def test_load_project_messages(self, db_with_messages):
        assert len(db_with_messages.messages) >= 2

    def test_lookup_full_name(self, db_with_messages):
        result = db_with_messages.lookup("test_pkg/msg/Position")
        assert result is not None
        assert result.name == "Position"

    def test_lookup_short_name(self, db_with_messages):
        result = db_with_messages.lookup("Position")
        assert result is not None
        assert result.name == "Position"

    def test_lookup_not_found(self, db_with_messages):
        result = db_with_messages.lookup("NonexistentMsg")
        assert result is None

    def test_search(self, db_with_messages):
        results = db_with_messages.search("vel")
        assert any(r.name == "Velocity" for r in results)

    def test_search_no_results(self, db_with_messages):
        results = db_with_messages.search("zzzznonexistent")
        assert len(results) == 0

    def test_get_used_messages(self, db_with_messages):
        used = db_with_messages.get_used_messages(["Position", "test_pkg/msg/Velocity"])
        assert len(used) >= 1

    def test_srv_lookup(self, db_with_messages):
        result = db_with_messages.lookup("GetState")
        assert result is not None
        assert result.kind == "srv"
        assert len(result.request_fields) == 1
        assert len(result.response_fields) == 2

    def test_to_dict(self, db_with_messages):
        d = db_with_messages.to_dict()
        assert isinstance(d, dict)
        assert len(d) > 0


class TestBuiltinTypes:
    """Tests for builtin type detection."""

    def test_all_builtins_present(self):
        expected = {"bool", "byte", "char", "float32", "float64",
                    "int8", "int16", "int32", "int64",
                    "uint8", "uint16", "uint32", "uint64",
                    "string", "wstring"}
        assert expected.issubset(BUILTIN_TYPES)


class TestLoadStandardMessages:
    """Tests for loading standard ROS2 messages."""

    def test_load_standard_messages(self):
        """Test that standard ROS2 messages can be loaded (if available)."""
        db = MessageDatabase()
        db.load_standard_messages()
        # On a system with ROS2 installed, we should find standard messages
        # On a system without ROS2, this should just return empty
        if len(db.messages) > 0:
            # Verify common types are loaded
            twist = db.lookup("geometry_msgs/msg/Twist")
            if twist:
                assert twist.name == "Twist"
                assert twist.package == "geometry_msgs"

            laser = db.lookup("sensor_msgs/msg/LaserScan")
            if laser:
                assert laser.name == "LaserScan"
                assert len(laser.fields) > 0


class TestLoadMessageDatabase:
    """Tests for the load_message_database convenience function."""

    def test_load_with_project(self, tmp_path):
        msg_dir = tmp_path / "test_pkg" / "msg"
        msg_dir.mkdir(parents=True)
        (msg_dir / "TestMsg.msg").write_text("float64 value\n")

        db = load_message_database(project_path=tmp_path, load_standard=False)
        assert len(db.messages) >= 1

    def test_load_without_standard(self, tmp_path):
        db = load_message_database(project_path=tmp_path, load_standard=False)
        # Should still work, just empty or with project messages only
        assert isinstance(db, MessageDatabase)
