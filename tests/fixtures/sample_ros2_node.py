"""
Sample ROS2 Node for Testing RoboMind Parser

This file demonstrates common ROS2 patterns that RoboMind should detect:
- Node class inheriting from rclpy.node.Node
- Publishers and subscribers
- Timers
- Parameters
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

# Global constants
LOOP_RATE = 10  # Hz
MAX_SPEED = 1.0
DEFAULT_TOPIC = "/cmd_vel"


class SampleNode(Node):
    """A sample ROS2 node demonstrating common patterns."""

    def __init__(self):
        super().__init__("sample_node")

        # Declare parameters
        self.declare_parameter("publish_rate", LOOP_RATE)
        self.declare_parameter("max_speed", MAX_SPEED)
        self.declare_parameter("topic_name", DEFAULT_TOPIC)

        # Get parameters
        self.publish_rate = self.get_parameter("publish_rate").value
        self.max_speed = self.get_parameter("max_speed").value
        topic_name = self.get_parameter("topic_name").value

        # Create publisher with literal topic name (statically analyzable)
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)

        # Note: A publisher using topic_name variable wouldn't be statically extractable
        # self.dynamic_publisher = self.create_publisher(Twist, topic_name, 10)

        # Create subscription
        self.subscription = self.create_subscription(
            String, "command", self.command_callback, 10
        )

        # Create timer for periodic publishing
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)

        self.command = "stop"
        self.get_logger().info("SampleNode initialized")

    def command_callback(self, msg):
        """Process incoming command."""
        self.command = msg.data
        self.get_logger().info(f"Received command: {self.command}")

    def timer_callback(self):
        """Periodic callback to publish velocity."""
        msg = Twist()

        if self.command == "forward":
            msg.linear.x = self.max_speed
        elif self.command == "backward":
            msg.linear.x = -self.max_speed
        elif self.command == "left":
            msg.angular.z = 1.0
        elif self.command == "right":
            msg.angular.z = -1.0
        # else: stop (all zeros)

        self.publisher.publish(msg)


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = SampleNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
