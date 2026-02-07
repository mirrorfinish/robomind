"""
RoboMind CLI - Command Line Interface

Usage:
    robomind scan <project_path>      # Scan for Python files
    robomind analyze <project_path>   # Full analysis with ROS2 extraction
    robomind launch <project_path>    # Analyze launch files and configs
    robomind visualize <project_path> # Generate interactive visualization
"""

import click
import json
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import print as rprint

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="robomind")
def main():
    """RoboMind - Rapid Prototyping System for Autonomous ROS2 Robots

    Analyze, visualize, and accelerate development of ROS2 robotics projects.
    """
    pass


@main.command()
@click.argument("project_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--output", "-o", type=click.Path(), help="Output JSON file path")
@click.option("--exclude", "-e", multiple=True,
              help="Glob patterns for paths to exclude (e.g., '*/archive/*')")
@click.option("--use-default-excludes/--no-default-excludes", default=True,
              help="Apply default excludes for archive/backup/test dirs (default: enabled)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def scan(project_path: str, output: Optional[str], exclude: tuple,
         use_default_excludes: bool, verbose: bool):
    """Scan a project directory for Python files and ROS2 packages.

    Use --exclude to filter out directories matching glob patterns:

    \b
    robomind scan ~/project --exclude "*/archive/*" --exclude "**/backup/**"

    By default, archive/backup/test directories are excluded. Use --no-default-excludes
    to include all files.

    PROJECT_PATH: Path to the robotics project to scan
    """
    from robomind.core.scanner import ProjectScanner

    project_path = Path(project_path).resolve()
    exclude_patterns = list(exclude) if exclude else None

    console.print(f"\n[bold blue]RoboMind Scanner[/bold blue]")
    console.print(f"Scanning: [cyan]{project_path}[/cyan]")
    if use_default_excludes:
        console.print(f"Default excludes: [green]enabled[/green] (archive, backup, test, etc.)")
    if exclude_patterns:
        console.print(f"Additional excludes: [yellow]{', '.join(exclude_patterns)}[/yellow]")
    console.print()

    try:
        scanner = ProjectScanner(
            project_path,
            exclude_patterns=exclude_patterns,
            use_default_excludes=use_default_excludes,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Scanning files...", total=None)
            result = scanner.scan()
            progress.update(task, completed=100, total=100)

        # Display results
        summary = result.summary()

        table = Table(title="Scan Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Root Path", str(summary["root"]))
        table.add_row("ROS2 Packages", str(summary["packages"]))
        table.add_row("Python Files", str(summary["python_files"]))
        table.add_row("Launch Files", str(summary["launch_files"]))
        table.add_row("Config Files", str(summary["config_files"]))
        table.add_row("Total Files", str(summary["total_files"]))
        table.add_row("Total Lines", f"{summary['total_lines']:,}")

        console.print(table)

        # Show packages found
        if result.packages:
            console.print(f"\n[bold]ROS2 Packages Found:[/bold]")
            for pkg_name, pkg_path in sorted(result.packages.items()):
                files = result.get_files_by_package(pkg_name)
                console.print(f"  [cyan]{pkg_name}[/cyan]: {len(files)} Python files")

        if verbose:
            console.print(f"\n[bold]Python Files:[/bold]")
            for pf in result.python_files[:20]:  # Show first 20
                console.print(f"  {pf.relative_path}")
            if len(result.python_files) > 20:
                console.print(f"  ... and {len(result.python_files) - 20} more")

        # Output to JSON if requested
        if output:
            output_path = Path(output)
            output_data = {
                "summary": summary,
                "packages": {k: str(v) for k, v in result.packages.items()},
                "python_files": [
                    {
                        "path": str(pf.path),
                        "relative_path": str(pf.relative_path),
                        "package": pf.package_name,
                        "size_bytes": pf.size_bytes,
                    }
                    for pf in result.python_files
                ],
                "launch_files": [str(lf.path) for lf in result.launch_files],
                "config_files": [str(cf.path) for cf in result.config_files],
            }

            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)

            console.print(f"\n[green]Output saved to: {output_path}[/green]")

        console.print(f"\n[bold green]Scan complete![/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@main.command()
@click.argument("project_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--output", "-o", type=click.Path(), default="robomind_analysis",
              help="Output directory for analysis results")
@click.option("--format", "-f", "formats", multiple=True,
              type=click.Choice(["json", "yaml", "html", "ai-context", "sarif"]),
              default=["json", "yaml", "html"],
              help="Output formats (can specify multiple)")
@click.option("--exclude", "-e", multiple=True,
              help="Glob patterns for paths to exclude (e.g., '*/archive/*')")
@click.option("--use-default-excludes/--no-default-excludes", default=True,
              help="Apply default excludes for archive/backup/test dirs (default: enabled)")
@click.option("--min-confidence", type=float, default=0.0,
              help="Minimum confidence threshold (0.0-1.0) to filter findings")
@click.option("--deployment-manifest", type=click.Path(exists=True),
              help="YAML file describing what actually runs in production")
@click.option("--trace-launch", type=click.Path(exists=True),
              help="Launch file to trace for deployed nodes")
@click.option("--http/--no-http", "detect_http", default=True,
              help="Detect HTTP/REST communication between services (default: enabled)")
@click.option("--remote", "-r", multiple=True,
              help="Remote hosts to analyze (user@host:path)")
@click.option("--key", "-k", type=click.Path(exists=True),
              help="SSH private key file for remote connections")
@click.option("--keep-remote", is_flag=True,
              help="Keep local copies of remote code after analysis")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def analyze(project_path: str, output: str, formats: tuple, exclude: tuple,
            use_default_excludes: bool, min_confidence: float,
            deployment_manifest: Optional[str], trace_launch: Optional[str],
            detect_http: bool, remote: tuple, key: Optional[str],
            keep_remote: bool, verbose: bool):
    """Perform full analysis of a ROS2 project.

    Extracts nodes, topics, parameters, and generates structured output.

    By default, archive/backup/test directories are excluded. Use --no-default-excludes
    to include all files.

    Use --min-confidence to filter out low-confidence findings (likely false positives):

    \b
    robomind analyze ~/betaray -o ./analysis --min-confidence 0.5

    Use --deployment-manifest to specify what code actually runs in production:

    \b
    robomind analyze ~/betaray --deployment-manifest deployment.yaml

    Use --trace-launch to trace a specific launch file for deployed nodes:

    \b
    robomind analyze ~/betaray --trace-launch src/bringup/launch/robot.launch.py

    For distributed systems, use --remote to analyze code on remote hosts:

    \b
    robomind analyze ~/local_project \\
        --remote robot@nav.local:~/betaray \\
        --remote robot@ai.local:~/betaray \\
        --key ~/.ssh/id_rsa

    PROJECT_PATH: Path to the robotics project to analyze
    """
    from robomind.core.scanner import ProjectScanner
    from robomind.core.parser import PythonParser
    from robomind.ros2.node_extractor import ROS2NodeExtractor
    from robomind.ros2.topic_extractor import TopicExtractor

    project_path = Path(project_path).resolve()
    output_dir = Path(output)
    key_file = Path(key) if key else None
    exclude_patterns = list(exclude) if exclude else None

    console.print(f"\n[bold blue]RoboMind Analyzer[/bold blue]")
    console.print(f"Project: [cyan]{project_path}[/cyan]")
    console.print(f"Output:  [cyan]{output_dir}[/cyan]")
    console.print(f"Formats: [cyan]{', '.join(formats)}[/cyan]")

    if use_default_excludes:
        console.print(f"Default excludes: [green]enabled[/green] (archive, backup, test, etc.)")
    if exclude_patterns:
        console.print(f"Additional excludes: [yellow]{', '.join(exclude_patterns)}[/yellow]")
    if min_confidence > 0:
        console.print(f"Min confidence: [cyan]{min_confidence}[/cyan]")
    if deployment_manifest:
        console.print(f"Deployment manifest: [cyan]{deployment_manifest}[/cyan]")
    if trace_launch:
        console.print(f"Trace launch: [cyan]{trace_launch}[/cyan]")

    if remote:
        console.print(f"Remote:  [cyan]{', '.join(remote)}[/cyan]")

    console.print()

    # Track all nodes from local and remote analysis
    all_nodes = []
    remote_results = None

    try:
        # Phase 0: Remote analysis (if specified)
        if remote:
            console.print("[bold]Phase 0: Analyzing remote hosts...[/bold]")
            from robomind.remote import DistributedAnalyzer, parse_remote_specs

            hosts = parse_remote_specs(list(remote), key_file)
            if not hosts:
                console.print("[yellow]  Warning: No valid remote hosts parsed[/yellow]")
            else:
                console.print(f"  Connecting to {len(hosts)} remote hosts...")

                # Test connections first
                analyzer = DistributedAnalyzer(hosts)
                connection_results = analyzer.test_connections()

                for hostname, connected in connection_results.items():
                    status = "[green]OK[/green]" if connected else "[red]FAILED[/red]"
                    console.print(f"    {hostname}: {status}")

                # Analyze connected hosts
                connected_hosts = [h for h in hosts if connection_results.get(h.hostname, False)]
                if connected_hosts:
                    remote_dir = output_dir / "remote_copies" if keep_remote else None
                    if remote_dir:
                        remote_dir.mkdir(parents=True, exist_ok=True)

                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                    ) as progress:
                        task = progress.add_task(
                            f"Analyzing {len(connected_hosts)} hosts...",
                            total=None
                        )
                        remote_analyzer = DistributedAnalyzer(connected_hosts)
                        remote_results = remote_analyzer.analyze_all(
                            output_dir=remote_dir,
                            keep_local_copies=keep_remote,
                        )
                        progress.update(task, completed=True)

                    # Report results
                    console.print(f"  Analyzed: {remote_results.hosts_analyzed} hosts")
                    if remote_results.hosts_failed > 0:
                        console.print(f"  [yellow]Failed: {remote_results.hosts_failed} hosts[/yellow]")
                        for error in remote_results.errors:
                            console.print(f"    [red]{error}[/red]")

                    # Add remote nodes to all_nodes
                    all_nodes.extend(remote_results.merged_nodes)
                    console.print(f"  Found {len(remote_results.merged_nodes)} ROS2 nodes from remote hosts")

        # Phase 1: Scan local project
        console.print("[bold]Phase 1: Scanning local project...[/bold]")
        scanner = ProjectScanner(
            project_path,
            exclude_patterns=exclude_patterns,
            use_default_excludes=use_default_excludes,
        )
        scan_result = scanner.scan()
        console.print(f"  Found {len(scan_result.python_files)} Python files, "
                     f"{len(scan_result.packages)} packages")

        # Phase 1.5: Deployment awareness (if manifest or launch tracing provided)
        deployment_info = None
        traced_nodes = set()  # Node names that are actually deployed
        external_dependencies = []  # External packages not in project

        if deployment_manifest or trace_launch:
            console.print("[bold]Phase 1.5: Analyzing deployment...[/bold]")

            if deployment_manifest:
                from robomind.deployment import load_deployment_manifest

                manifest = load_deployment_manifest(Path(deployment_manifest))
                deployment_info = manifest
                console.print(f"  Manifest: {manifest.summary()['total_hosts']} hosts, "
                             f"{manifest.summary()['launch_files']} launch files")

                # Track deployed launch files
                for lf in manifest.deployed_launch_files:
                    console.print(f"    Launch: [cyan]{lf}[/cyan]")

            if trace_launch:
                from robomind.deployment import trace_launch_file

                trace = trace_launch_file(Path(trace_launch), project_root=project_path)
                console.print(f"  Traced {trace.summary()['total_nodes']} nodes from launch file")

                # Get project packages for comparison (packages is a dict with names as keys)
                project_packages = set(scan_result.packages.keys())

                # Add traced node names and detect external dependencies
                for node in trace.nodes:
                    traced_nodes.add(node.name)
                    traced_nodes.add(node.executable)

                    # Check if package is external (not in project)
                    if node.package and node.package not in project_packages:
                        ext_info = {
                            "package": node.package,
                            "executable": node.executable,
                            "name": node.name,
                        }
                        # Avoid duplicates
                        if not any(e["package"] == node.package and e["executable"] == node.executable
                                   for e in external_dependencies):
                            external_dependencies.append(ext_info)

                if external_dependencies:
                    console.print(f"  [cyan]External dependencies: {len(external_dependencies)}[/cyan]")
                    for ext in external_dependencies:
                        console.print(f"    - {ext['package']}/{ext['executable']}")

                if trace.errors:
                    for error in trace.errors[:3]:
                        console.print(f"    [yellow]Warning: {error}[/yellow]")

        # Phase 2: Parse
        console.print("[bold]Phase 2: Parsing Python files...[/bold]")
        parser = PythonParser()
        parse_results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Parsing...", total=len(scan_result.python_files))

            for pf in scan_result.python_files:
                result = parser.parse_file(pf.path)
                if result:
                    result.package_name = pf.package_name
                    parse_results.append(result)
                progress.advance(task)

        # Count statistics
        total_classes = sum(len(pr.classes) for pr in parse_results)
        total_functions = sum(len(pr.functions) for pr in parse_results)

        console.print(f"  Parsed {len(parse_results)} files: "
                     f"{total_classes} classes, {total_functions} functions")

        # Phase 3: ROS2 Extraction (local)
        console.print("[bold]Phase 3: Extracting ROS2 constructs (local)...[/bold]")

        node_extractor = ROS2NodeExtractor()
        topic_extractor = TopicExtractor()
        local_nodes = []

        # Filter to files with ROS2 imports for efficiency
        ros2_files = [pr for pr in parse_results if pr.has_ros2_imports()]
        console.print(f"  Found {len(ros2_files)} files with ROS2 imports")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting nodes...", total=len(ros2_files))

            for pr in ros2_files:
                nodes = node_extractor.extract_from_file(pr.file_path, pr.package_name)
                local_nodes.extend(nodes)
                progress.advance(task)

        # Combine local and remote nodes
        all_nodes.extend(local_nodes)

        # Build unified topic graph with all nodes
        for node in all_nodes:
            topic_extractor.add_nodes([node])

        topic_graph = topic_extractor.build()

        # Statistics (before filtering)
        total_nodes_before = len(all_nodes)
        total_publishers = sum(len(n.publishers) for n in all_nodes)
        total_subscribers = sum(len(n.subscribers) for n in all_nodes)
        total_timers = sum(len(n.timers) for n in all_nodes)
        total_params = sum(len(n.parameters) for n in all_nodes)

        console.print(f"  Found {len(all_nodes)} ROS2 nodes")
        console.print(f"  Publishers: {total_publishers}, Subscribers: {total_subscribers}")
        console.print(f"  Timers: {total_timers}, Parameters: {total_params}")
        console.print(f"  Topics: {len(topic_graph.topics)} "
                     f"({len(topic_graph.get_connected_topics())} connected)")

        # Phase 3.6: Filter to deployed nodes (if --trace-launch provided)
        if traced_nodes:
            console.print("[bold]Phase 3.6: Filtering to deployed nodes...[/bold]")

            def normalize(name: str) -> str:
                """Normalize name for matching: lowercase, remove underscores/hyphens."""
                return name.lower().replace("_", "").replace("-", "")

            # Match nodes by name, class_name, executable, or file name
            def is_deployed(node):
                # Get all name forms for this node
                node_names = [
                    node.name,
                    node.class_name or "",
                    # Extract filename without extension
                    Path(node.file_path).stem if node.file_path else "",
                ]
                node_names_normalized = [normalize(n) for n in node_names if n]

                for traced in traced_nodes:
                    traced_norm = normalize(traced)

                    # Check all node name forms
                    for node_norm in node_names_normalized:
                        # Direct match
                        if traced_norm == node_norm:
                            return True
                        # Partial match (e.g., "motor_controller" matches "betaray_motor_controller")
                        if traced_norm in node_norm or node_norm in traced_norm:
                            return True

                return False

            deployed_nodes = [n for n in all_nodes if is_deployed(n)]
            filtered_out = len(all_nodes) - len(deployed_nodes)

            console.print(f"  Traced {len(traced_nodes)} nodes from launch file")
            console.print(f"  Deployed: {len(deployed_nodes)} / {len(all_nodes)} nodes")
            console.print(f"  [yellow]Filtered out: {filtered_out} nodes (not in launch file)[/yellow]")

            # Replace all_nodes with deployed nodes only
            all_nodes = deployed_nodes

            # Rebuild topic graph with filtered nodes
            topic_extractor = TopicExtractor()
            for node in all_nodes:
                topic_extractor.add_nodes([node])
            topic_graph = topic_extractor.build()

            console.print(f"  Topics after filter: {len(topic_graph.topics)} "
                         f"({len(topic_graph.get_connected_topics())} connected)")

        # Phase 3.5: Confidence scoring (if min_confidence > 0)
        if min_confidence > 0:
            console.print("[bold]Phase 3.5: Calculating confidence scores...[/bold]")
            from robomind.analyzers.confidence import (
                ConfidenceCalculator,
                NodeConfidenceContext,
                get_confidence_summary,
            )
            from robomind.ros2.launch_analyzer import LaunchFileAnalyzer

            # Build lookup of which nodes are in launch files
            launch_analyzer = LaunchFileAnalyzer()
            node_to_launch_files = {}

            for lf in scan_result.launch_files:
                launch_info = launch_analyzer.analyze_file(lf.path)
                for node_decl in launch_info.nodes:
                    key = node_decl.name
                    if key not in node_to_launch_files:
                        node_to_launch_files[key] = []
                    node_to_launch_files[key].append(str(lf.relative_path))

            # Add nodes from deployment manifest launch files if available
            if deployment_info:
                for lf_name in deployment_info.deployed_launch_files:
                    # Find this launch file in scan results
                    for lf in scan_result.launch_files:
                        if Path(lf_name).name in str(lf.relative_path):
                            launch_info = launch_analyzer.analyze_file(lf.path)
                            for node_decl in launch_info.nodes:
                                if node_decl.name not in node_to_launch_files:
                                    node_to_launch_files[node_decl.name] = []
                                if str(lf.relative_path) not in node_to_launch_files[node_decl.name]:
                                    node_to_launch_files[node_decl.name].append(str(lf.relative_path))
                            break

            # Build lookup for file location confidence
            file_confidence_map = {
                str(pf.path): (pf.location_confidence, pf.dead_code_indicators)
                for pf in scan_result.python_files
            }

            # Calculate confidence for each node
            calculator = ConfidenceCalculator(min_confidence=min_confidence)
            node_scores = []

            for node in all_nodes:
                # Get file-based confidence info
                loc_conf, dead_indicators = file_confidence_map.get(
                    str(node.file_path), (1.0, [])
                )

                # Determine launch files for this node
                node_launch_files = node_to_launch_files.get(node.name, [])

                # If traced_nodes has entries and this node is traced, add synthetic entry
                if traced_nodes and (node.name in traced_nodes or
                                     node.class_name in traced_nodes):
                    if not node_launch_files:
                        node_launch_files = ["[traced from --trace-launch]"]

                # Build context
                context = NodeConfidenceContext(
                    node_name=node.name,
                    file_path=node.file_path,
                    package_name=node.package_name,
                    location_confidence=loc_conf,
                    dead_code_indicators=dead_indicators,
                    in_launch_files=node_launch_files,
                    has_publishers=len(node.publishers) > 0,
                    has_subscribers=len(node.subscribers) > 0,
                    publisher_topics=[p.topic for p in node.publishers],
                    subscriber_topics=[s.topic for s in node.subscribers],
                    topics_with_both_pubsub=[
                        t for t in topic_graph.get_connected_topics()
                        if any(p.topic == t for p in node.publishers) or
                           any(s.topic == t for s in node.subscribers)
                    ],
                )

                score = calculator.calculate_node_confidence(context)
                node_scores.append((node, score))

            # Filter by minimum confidence
            filtered = [(n, s) for n, s in node_scores if s.score >= min_confidence]
            all_nodes = [n for n, s in filtered]

            # Report confidence summary
            all_scores = [s for n, s in node_scores]
            summary = get_confidence_summary(all_scores)

            console.print(f"  Confidence: avg {summary['average']:.2f}, "
                         f"{summary['above_0.7']} high, {summary['below_0.3']} low")
            console.print(f"  Filtered: {total_nodes_before} -> {len(all_nodes)} nodes "
                         f"(min_confidence={min_confidence})")

            # Rebuild topic graph with filtered nodes
            topic_extractor = TopicExtractor()
            for node in all_nodes:
                topic_extractor.add_nodes([node])
            topic_graph = topic_extractor.build()

        # Phase 3.7: HTTP communication detection
        http_comm_map = None
        if detect_http:
            console.print("[bold]Phase 3.7: Detecting HTTP communication...[/bold]")
            from robomind.http import (
                HTTPEndpointExtractor,
                HTTPClientExtractor,
                build_communication_map,
            )

            endpoint_extractor = HTTPEndpointExtractor()
            client_extractor = HTTPClientExtractor()

            all_endpoints = []
            all_clients = []

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console,
            ) as progress:
                task = progress.add_task("Scanning for HTTP...", total=len(scan_result.python_files))

                for pf in scan_result.python_files:
                    # Extract endpoints (Flask/FastAPI servers)
                    endpoints = endpoint_extractor.extract_from_file(pf.path)
                    all_endpoints.extend(endpoints)

                    # Extract clients (requests, httpx, aiohttp)
                    clients = client_extractor.extract_from_file(pf.path)
                    all_clients.extend(clients)

                    progress.advance(task)

            # Build communication map
            http_comm_map = build_communication_map(
                http_endpoints=all_endpoints,
                http_clients=all_clients,
                ros2_topic_graph=topic_graph,
            )

            # Report findings
            summary = http_comm_map.summary()
            console.print(f"  HTTP endpoints: {summary['http_endpoints']}")
            console.print(f"  HTTP clients: {summary['http_clients']}")
            if summary['http_target_hosts']:
                console.print(f"  Target hosts: {', '.join(summary['http_target_hosts'][:5])}")
            console.print(f"  Communication links: {summary['total_links']} "
                         f"({summary['http_links']} HTTP, {summary['ros2_links']} ROS2)")

        # Phase 3.8: AI service detection
        ai_analysis_result = None
        console.print("[bold]Phase 3.8: Detecting AI/ML services...[/bold]")
        from robomind.analyzers.ai_service_analyzer import AIServiceAnalyzer

        ai_analyzer = AIServiceAnalyzer()
        ai_analysis_result = ai_analyzer.analyze_files(
            [pf.path for pf in scan_result.python_files]
        )

        if ai_analysis_result.services:
            console.print(f"  AI services: {len(ai_analysis_result.services)}")
            for svc in ai_analysis_result.services:
                port_str = f" :{svc.port}" if svc.port else ""
                model_str = f" ({svc.model_name})" if svc.model_name else ""
                console.print(f"    {svc.framework}{port_str}{model_str} [{len(svc.caller_files)} callers]")
        else:
            console.print("  No AI services detected")

        # Phase 3.9: Parse message definitions
        message_defs_dict = None
        console.print("[bold]Phase 3.9: Parsing message definitions...[/bold]")
        try:
            from robomind.ros2.message_parser import load_message_database
            msg_db = load_message_database(project_path=project_path, load_standard=True)

            # Collect all message types used by detected nodes
            used_types = set()
            for node in all_nodes:
                for pub in node.publishers:
                    if pub.msg_type:
                        used_types.add(pub.msg_type)
                for sub in node.subscribers:
                    if sub.msg_type:
                        used_types.add(sub.msg_type)
                for svc in node.services:
                    if svc.srv_type:
                        used_types.add(svc.srv_type)

            # Filter to only used types
            used_messages = msg_db.get_used_messages(list(used_types))
            if used_messages:
                message_defs_dict = {k: v.to_dict() for k, v in used_messages.items()}
                console.print(f"  Resolved {len(used_messages)} message schemas "
                             f"(from {len(msg_db.messages)} available)")
            else:
                console.print("  No message types resolved")
        except Exception as e:
            console.print(f"  [yellow]Message parsing skipped:[/yellow] {e}")

        # Phase 4: Build graph and coupling (for exports)
        console.print("[bold]Phase 4: Building system graph...[/bold]")
        from robomind.core.graph import build_system_graph, GraphBuilder
        from robomind.analyzers.coupling import analyze_coupling

        system_graph = build_system_graph(all_nodes, topic_graph)

        # Add AI services to the graph
        if ai_analysis_result and ai_analysis_result.services:
            builder = GraphBuilder()
            builder.graph = system_graph
            for svc in ai_analysis_result.services:
                builder.add_ai_service(
                    name=svc.name,
                    framework=svc.framework,
                    port=svc.port,
                    model_name=svc.model_name,
                    endpoint_path=svc.endpoint_path,
                    file_path=svc.file_path,
                    line_number=svc.line_number,
                    gpu_required=svc.gpu_required,
                )
        coupling_matrix = analyze_coupling(all_nodes, topic_graph)

        graph_stats = system_graph.stats()
        coupling_summary = coupling_matrix.summary()
        console.print(f"  Graph: {graph_stats['total_nodes']} nodes, {graph_stats['total_edges']} edges")
        console.print(f"  Coupling: {coupling_summary['total_pairs']} pairs, "
                     f"{coupling_summary['critical_pairs']} critical")

        # Phase 5: Export
        console.print("[bold]Phase 5: Generating output...[/bold]")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON export
        if "json" in formats:
            from robomind.exporters.json_exporter import export_analysis_json

            json_path = output_dir / "system_graph.json"
            result = export_analysis_json(
                output_path=json_path,
                nodes=all_nodes,
                system_graph=system_graph,
                coupling=coupling_matrix,
                topic_graph=topic_graph,
                project_name=project_path.name,
                project_path=str(project_path),
                http_comm_map=http_comm_map,
                external_dependencies=external_dependencies if external_dependencies else None,
                ai_services=ai_analysis_result,
                message_definitions=message_defs_dict,
            )

            if result.success:
                console.print(f"  [green]JSON:[/green] {json_path}")
            else:
                console.print(f"  [red]JSON failed:[/red] {result.error}")

        # YAML export
        if "yaml" in formats:
            from robomind.exporters.yaml_exporter import export_yaml_context

            results = export_yaml_context(
                output_dir=output_dir,
                nodes=all_nodes,
                system_graph=system_graph,
                coupling=coupling_matrix,
                topic_graph=topic_graph,
                project_name=project_path.name,
                http_comm_map=http_comm_map,
                ai_services=ai_analysis_result,
            )

            for name, result in results.items():
                if result.success:
                    console.print(f"  [green]YAML ({name}):[/green] {result.output_path} "
                                 f"(~{result.token_estimate} tokens)")
                else:
                    console.print(f"  [red]YAML ({name}) failed:[/red] {result.error}")

        if "html" in formats:
            from robomind.exporters.html_exporter import export_html_visualization

            html_path = output_dir / "visualization.html"
            result = export_html_visualization(
                output_path=html_path,
                system_graph=system_graph,
                project_name=project_path.name,
                coupling=coupling_matrix,
                nodes=all_nodes,
                topic_graph=topic_graph,
                http_comm_map=http_comm_map,
            )

            if result.success:
                console.print(f"  [green]HTML:[/green] {html_path}")
            else:
                console.print(f"  [red]HTML failed:[/red] {result.error}")

        if "ai-context" in formats:
            from robomind.exporters.ai_context_exporter import export_ai_context

            ai_path = output_dir / "ai_context.yaml"
            # Build confidence scores dict if available
            node_confidence = {}
            if min_confidence > 0 and 'node_scores' in dir():
                node_confidence = {n.name: s.score for n, s in node_scores}

            success = export_ai_context(
                output_path=ai_path,
                nodes=all_nodes,
                topic_graph=topic_graph,
                coupling=coupling_matrix,
                http_comm_map=http_comm_map,
                confidence_scores=node_confidence,
                project_name=project_path.name,
            )

            if success:
                console.print(f"  [green]AI Context:[/green] {ai_path}")
            else:
                console.print(f"  [red]AI Context failed[/red]")

        if "sarif" in formats:
            from robomind.exporters.sarif_exporter import export_sarif

            sarif_path = output_dir / "robomind.sarif"
            # Build confidence scores dict if available
            node_confidence = {}
            if min_confidence > 0 and 'node_scores' in dir():
                node_confidence = {n.name: s.score for n, s in node_scores}

            success = export_sarif(
                output_path=sarif_path,
                nodes=all_nodes,
                topic_graph=topic_graph,
                coupling=coupling_matrix,
                confidence_scores=node_confidence,
                project_name=project_path.name,
                project_path=str(project_path),
            )

            if success:
                console.print(f"  [green]SARIF:[/green] {sarif_path}")
            else:
                console.print(f"  [red]SARIF failed[/red]")

        console.print(f"\n[bold green]Analysis complete![/bold green]")
        console.print(f"Results saved to: [cyan]{output_dir}[/cyan]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument("project_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--output", "-o", type=click.Path(), help="Output JSON file path")
@click.option("--exclude", "-e", multiple=True,
              help="Glob patterns for paths to exclude (e.g., '*/archive/*')")
@click.option("--use-default-excludes/--no-default-excludes", default=True,
              help="Apply default excludes for archive/backup/test dirs (default: enabled)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def launch(project_path: str, output: Optional[str], exclude: tuple,
           use_default_excludes: bool, verbose: bool):
    """Analyze ROS2 launch files and parameter configs.

    Extracts launch topology, node sequences, and parameter configurations.

    PROJECT_PATH: Path to the robotics project to analyze
    """
    from robomind.core.scanner import ProjectScanner
    from robomind.ros2.launch_analyzer import LaunchFileAnalyzer, LaunchTopology
    from robomind.ros2.param_extractor import ConfigScanner

    project_path = Path(project_path).resolve()
    exclude_patterns = list(exclude) if exclude else None

    console.print(f"\n[bold blue]RoboMind Launch Analyzer[/bold blue]")
    console.print(f"Project: [cyan]{project_path}[/cyan]")
    if use_default_excludes:
        console.print(f"Default excludes: [green]enabled[/green]")
    if exclude_patterns:
        console.print(f"Additional excludes: [yellow]{', '.join(exclude_patterns)}[/yellow]")
    console.print()

    try:
        # Phase 1: Find launch files
        console.print("[bold]Phase 1: Scanning for launch files...[/bold]")
        scanner = ProjectScanner(
            project_path,
            exclude_patterns=exclude_patterns,
            use_default_excludes=use_default_excludes,
        )
        scan_result = scanner.scan()

        launch_files = scan_result.launch_files
        console.print(f"  Found {len(launch_files)} launch files")

        # Phase 2: Analyze launch files
        console.print("[bold]Phase 2: Parsing launch files...[/bold]")
        analyzer = LaunchFileAnalyzer()
        topology = LaunchTopology()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing...", total=len(launch_files))

            for lf in launch_files:
                info = analyzer.analyze_file(lf.path)
                if info.total_nodes > 0 or info.arguments:
                    topology.launch_files.append(info)
                progress.advance(task)

        # Phase 3: Scan config files
        console.print("[bold]Phase 3: Scanning parameter configs...[/bold]")
        config_scanner = ConfigScanner(project_path)
        param_collection = config_scanner.scan()

        console.print(f"  Found {len(param_collection.files)} config files, "
                     f"{param_collection.summary()['total_parameters']} parameters")

        # Display results
        console.print()

        # Launch file summary
        table = Table(title="Launch File Analysis")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        topo_summary = topology.summary()
        table.add_row("Launch Files Parsed", str(len(topology.launch_files)))
        table.add_row("Total Nodes", str(topo_summary["total_nodes"]))
        table.add_row("Composable Nodes", str(topo_summary["composable_nodes"]))
        table.add_row("Conditional Nodes", str(topo_summary["conditional_nodes"]))
        table.add_row("Launch Arguments", str(topo_summary["total_arguments"]))
        table.add_row("Unique Packages", str(topo_summary["unique_packages"]))
        table.add_row("Max Delay", f"{topo_summary['max_delay']:.1f}s")

        console.print(table)

        # Show launch files with nodes
        if topology.launch_files:
            console.print(f"\n[bold]Launch Files:[/bold]")
            for lf in sorted(topology.launch_files, key=lambda x: x.total_nodes, reverse=True):
                summary = lf.summary()
                console.print(f"  [cyan]{lf.file_path.name}[/cyan]: "
                             f"{summary['nodes']} nodes, {summary['arguments']} args, "
                             f"max delay {summary['max_delay']:.1f}s")

                if verbose:
                    # Show launch sequence
                    for item in lf.get_launch_sequence()[:10]:
                        delay_str = f"[+{item['delay']:.1f}s]" if item['delay'] > 0 else "[0s]"
                        cond = ""
                        if hasattr(item['item'], 'condition') and item['item'].condition:
                            cond = f" [dim](conditional)[/dim]"
                        console.print(f"    {delay_str} {item['package']}/{item['name']}{cond}")
                    if len(lf.get_launch_sequence()) > 10:
                        console.print(f"    ... and {len(lf.get_launch_sequence()) - 10} more")

        # Show config files
        if param_collection.files:
            console.print(f"\n[bold]Parameter Config Files:[/bold]")
            for cf in sorted(param_collection.files, key=lambda x: x.total_parameters, reverse=True)[:10]:
                console.print(f"  [cyan]{cf.file_path.name}[/cyan]: "
                             f"{cf.total_parameters} params for "
                             f"{', '.join(n.node_name for n in cf.nodes)}")
            if len(param_collection.files) > 10:
                console.print(f"  ... and {len(param_collection.files) - 10} more")

        # Output to JSON if requested
        if output:
            output_path = Path(output)
            output_data = {
                "launch_topology": topology.to_dict(),
                "parameters": param_collection.to_dict(),
            }

            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2, default=str)

            console.print(f"\n[green]Output saved to: {output_path}[/green]")

        console.print(f"\n[bold green]Launch analysis complete![/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument("project_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--output", "-o", type=click.Path(), help="Output JSON file path")
@click.option("--graphml", type=click.Path(), help="Export GraphML file path")
@click.option("--exclude", "-e", multiple=True,
              help="Glob patterns for paths to exclude (e.g., '*/archive/*')")
@click.option("--use-default-excludes/--no-default-excludes", default=True,
              help="Apply default excludes for archive/backup/test dirs (default: enabled)")
@click.option("--coupling/--no-coupling", default=True, help="Include coupling analysis")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def graph(project_path: str, output: Optional[str], graphml: Optional[str],
          exclude: tuple, use_default_excludes: bool, coupling: bool, verbose: bool):
    """Build and analyze the system dependency graph.

    Creates a NetworkX-based graph of ROS2 nodes, topics, services,
    and their relationships. Includes optional coupling analysis.

    PROJECT_PATH: Path to the robotics project to analyze
    """
    from robomind.core.scanner import ProjectScanner
    from robomind.core.parser import PythonParser
    from robomind.ros2.node_extractor import ROS2NodeExtractor
    from robomind.ros2.topic_extractor import TopicExtractor
    from robomind.core.graph import build_system_graph
    from robomind.analyzers.coupling import CouplingAnalyzer

    project_path = Path(project_path).resolve()
    exclude_patterns = list(exclude) if exclude else None

    console.print(f"\n[bold blue]RoboMind Graph Builder[/bold blue]")
    console.print(f"Project: [cyan]{project_path}[/cyan]")
    if use_default_excludes:
        console.print(f"Default excludes: [green]enabled[/green]")
    if exclude_patterns:
        console.print(f"Additional excludes: [yellow]{', '.join(exclude_patterns)}[/yellow]")
    console.print()

    try:
        # Phase 1: Scan and extract
        console.print("[bold]Phase 1: Extracting ROS2 nodes...[/bold]")
        scanner = ProjectScanner(
            project_path,
            exclude_patterns=exclude_patterns,
            use_default_excludes=use_default_excludes,
        )
        scan_result = scanner.scan()

        parser = PythonParser()
        node_extractor = ROS2NodeExtractor()
        topic_extractor = TopicExtractor()
        all_nodes = []

        # Parse and extract
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting...", total=len(scan_result.python_files))

            for pf in scan_result.python_files:
                pr = parser.parse_file(pf.path)
                if pr and pr.has_ros2_imports():
                    nodes = node_extractor.extract_from_file(pf.path, pf.package_name)
                    all_nodes.extend(nodes)
                    topic_extractor.add_nodes(nodes)
                progress.advance(task)

        topic_graph = topic_extractor.build()
        console.print(f"  Found {len(all_nodes)} ROS2 nodes, {len(topic_graph.topics)} topics")

        # Phase 2: Build graph
        console.print("[bold]Phase 2: Building system graph...[/bold]")
        system_graph = build_system_graph(all_nodes, topic_graph)

        stats = system_graph.stats()
        console.print(f"  Graph: {stats['total_nodes']} nodes, {stats['total_edges']} edges")

        # Display graph stats
        table = Table(title="System Graph Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Nodes", str(stats['total_nodes']))
        table.add_row("Total Edges", str(stats['total_edges']))
        table.add_row("Is DAG", "Yes" if stats['is_dag'] else "No")
        table.add_row("Connected Components", str(stats['weakly_connected_components']))

        for node_type, count in stats.get('node_types', {}).items():
            table.add_row(f"  {node_type}", str(count))

        console.print(table)

        # Phase 3: Coupling analysis (if enabled)
        coupling_matrix = None
        if coupling and len(all_nodes) > 1:
            console.print("[bold]Phase 3: Analyzing coupling...[/bold]")
            analyzer = CouplingAnalyzer(all_nodes, topic_graph)
            coupling_matrix = analyzer.analyze()

            summary = coupling_matrix.summary()

            coupling_table = Table(title="Coupling Analysis")
            coupling_table.add_column("Metric", style="cyan")
            coupling_table.add_column("Value", style="green")

            coupling_table.add_row("Connected Pairs", str(summary['total_pairs']))
            coupling_table.add_row("Average Coupling", f"{summary['average_coupling']:.3f}")
            coupling_table.add_row("Critical Pairs", str(summary['critical_pairs']))
            coupling_table.add_row("High Pairs", str(summary['high_pairs']))
            coupling_table.add_row("Medium Pairs", str(summary['medium_pairs']))
            coupling_table.add_row("Low Pairs", str(summary['low_pairs']))

            console.print(coupling_table)

            # Show top coupled pairs
            if verbose:
                top_pairs = coupling_matrix.get_top_coupled_pairs(10)
                if top_pairs:
                    console.print("\n[bold]Top Coupled Pairs:[/bold]")
                    for score in top_pairs:
                        strength_color = {
                            "CRITICAL": "red",
                            "HIGH": "yellow",
                            "MEDIUM": "cyan",
                            "LOW": "green",
                        }.get(score.strength, "white")
                        console.print(f"  {score.source} <-> {score.target}: "
                                     f"[{strength_color}]{score.score:.3f} ({score.strength})[/{strength_color}]")
                        console.print(f"    Topics: {', '.join(score.topics[:3])}"
                                     f"{'...' if len(score.topics) > 3 else ''}")

        # Critical nodes
        if verbose:
            console.print("\n[bold]Critical Nodes (by centrality):[/bold]")
            for node_id, centrality in system_graph.get_critical_nodes(10):
                node = system_graph.get_node(node_id)
                if node and centrality > 0:
                    console.print(f"  {node.name} ({node.component_type.name}): {centrality:.4f}")

        # Cycle detection
        cycles = system_graph.find_cycles()
        if cycles:
            console.print(f"\n[yellow]Warning: {len(cycles)} cycles detected in graph[/yellow]")
            if verbose:
                for cycle in cycles[:5]:
                    console.print(f"  {' -> '.join(cycle[:5])}{'...' if len(cycle) > 5 else ''}")

        # Export outputs
        if output:
            output_path = Path(output)
            export_data = {
                "graph": system_graph.to_dict(),
            }
            if coupling_matrix:
                export_data["coupling"] = coupling_matrix.to_dict()

            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2, default=str)
            console.print(f"\n[green]JSON saved to: {output_path}[/green]")

        if graphml:
            graphml_path = Path(graphml)
            system_graph.export_graphml(graphml_path)
            console.print(f"[green]GraphML saved to: {graphml_path}[/green]")

        console.print(f"\n[bold green]Graph analysis complete![/bold green]")

    except ImportError as e:
        if "networkx" in str(e).lower():
            console.print("[bold red]Error:[/bold red] NetworkX is required for graph analysis.")
            console.print("Install with: pip install networkx")
        else:
            raise
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument("project_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--output", "-o", type=click.Path(), default="visualization.html",
              help="Output HTML file path")
@click.option("--exclude", "-e", multiple=True,
              help="Glob patterns for paths to exclude (e.g., '*/archive/*')")
@click.option("--use-default-excludes/--no-default-excludes", default=True,
              help="Apply default excludes for archive/backup/test dirs (default: enabled)")
@click.option("--open", "open_browser", is_flag=True, help="Open in browser after generation")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def visualize(project_path: str, output: str, exclude: tuple,
              use_default_excludes: bool, open_browser: bool, verbose: bool):
    """Generate interactive D3.js visualization of the project.

    Creates a standalone HTML file with:
    - Force-directed graph layout
    - Color-coded nodes by type
    - Interactive zoom/pan
    - Search and filter functionality

    PROJECT_PATH: Path to the robotics project to visualize
    """
    from robomind.core.scanner import ProjectScanner
    from robomind.core.parser import PythonParser
    from robomind.ros2.node_extractor import ROS2NodeExtractor
    from robomind.ros2.topic_extractor import TopicExtractor
    from robomind.core.graph import build_system_graph
    from robomind.analyzers.coupling import analyze_coupling
    from robomind.exporters.html_exporter import export_html_visualization

    project_path = Path(project_path).resolve()
    output_path = Path(output)
    exclude_patterns = list(exclude) if exclude else None

    console.print(f"\n[bold blue]RoboMind Visualizer[/bold blue]")
    console.print(f"Project: [cyan]{project_path}[/cyan]")
    console.print(f"Output:  [cyan]{output_path}[/cyan]")
    if use_default_excludes:
        console.print(f"Default excludes: [green]enabled[/green]")
    if exclude_patterns:
        console.print(f"Additional excludes: [yellow]{', '.join(exclude_patterns)}[/yellow]")
    console.print()

    try:
        # Phase 1: Extract ROS2 nodes
        console.print("[bold]Phase 1: Extracting ROS2 nodes...[/bold]")
        scanner = ProjectScanner(
            project_path,
            exclude_patterns=exclude_patterns,
            use_default_excludes=use_default_excludes,
        )
        scan_result = scanner.scan()

        parser = PythonParser()
        node_extractor = ROS2NodeExtractor()
        topic_extractor = TopicExtractor()
        all_nodes = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting...", total=len(scan_result.python_files))

            for pf in scan_result.python_files:
                pr = parser.parse_file(pf.path)
                if pr and pr.has_ros2_imports():
                    nodes = node_extractor.extract_from_file(pf.path, pf.package_name)
                    all_nodes.extend(nodes)
                    topic_extractor.add_nodes(nodes)
                progress.advance(task)

        topic_graph = topic_extractor.build()
        console.print(f"  Found {len(all_nodes)} ROS2 nodes, {len(topic_graph.topics)} topics")

        # Phase 2: Build graph
        console.print("[bold]Phase 2: Building system graph...[/bold]")
        system_graph = build_system_graph(all_nodes, topic_graph)
        coupling_matrix = analyze_coupling(all_nodes, topic_graph)

        stats = system_graph.stats()
        console.print(f"  Graph: {stats['total_nodes']} nodes, {stats['total_edges']} edges")

        # Phase 3: Generate visualization
        console.print("[bold]Phase 3: Generating visualization...[/bold]")

        result = export_html_visualization(
            output_path=output_path,
            system_graph=system_graph,
            project_name=project_path.name,
            coupling=coupling_matrix,
            nodes=all_nodes,
            topic_graph=topic_graph,
            open_browser=open_browser,
        )

        if result.success:
            console.print(f"  [green]Generated:[/green] {result.output_path}")
            console.print(f"  File size: {result.stats.get('file_size', 0):,} bytes")
            console.print(f"  Nodes: {result.stats.get('nodes', 0)}, Edges: {result.stats.get('edges', 0)}")

            if open_browser:
                console.print("  [cyan]Opening in browser...[/cyan]")

            console.print(f"\n[bold green]Visualization complete![/bold green]")
        else:
            console.print(f"[bold red]Error:[/bold red] {result.error}")
            sys.exit(1)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument("hosts", nargs=-1, required=True)
@click.option("--key", "-k", type=click.Path(exists=True),
              help="SSH private key file")
@click.option("--ros2-info", is_flag=True,
              help="Get live ROS2 system info from remote")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def remote(hosts: tuple, key: Optional[str], ros2_info: bool, verbose: bool):
    """Test connections and get info from remote hosts.

    HOSTS: Remote host specifications (user@host or user@host:path)

    \b
    Examples:
        robomind remote robot@nav.local robot@ai.local
        robomind remote robot@jetson:~/betaray --ros2-info
        robomind remote robot@nav.local --key ~/.ssh/id_rsa
    """
    from robomind.remote import parse_remote_specs, SSHAnalyzer

    key_file = Path(key) if key else None
    parsed_hosts = parse_remote_specs(list(hosts), key_file)

    if not parsed_hosts:
        console.print("[bold red]Error:[/bold red] No valid hosts specified")
        sys.exit(1)

    console.print(f"\n[bold blue]RoboMind Remote Connection Test[/bold blue]")
    console.print(f"Testing {len(parsed_hosts)} hosts...\n")

    table = Table(title="Remote Host Status")
    table.add_column("Host", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Path", style="dim")

    if ros2_info:
        table.add_column("ROS2 Nodes")
        table.add_column("Topics")

    all_connected = True
    for host in parsed_hosts:
        analyzer = SSHAnalyzer(host)
        connected = analyzer.connect()

        if connected:
            status = "[green]Connected[/green]"
            row = [host.hostname, status, host.project_path]

            if ros2_info:
                info = analyzer.get_remote_ros2_info()
                row.append(str(len(info.get("nodes", []))))
                row.append(str(len(info.get("topics", []))))

                if verbose:
                    console.print(f"\n[bold]{host.hostname}[/bold] ROS2 Info:")
                    if info["nodes"]:
                        console.print("  Nodes:")
                        for node in info["nodes"][:10]:
                            console.print(f"    {node}")
                        if len(info["nodes"]) > 10:
                            console.print(f"    ... and {len(info['nodes']) - 10} more")
                    if info["topics"]:
                        console.print("  Topics:")
                        for topic in info["topics"][:10]:
                            console.print(f"    {topic}")
                        if len(info["topics"]) > 10:
                            console.print(f"    ... and {len(info['topics']) - 10} more")
        else:
            status = "[red]Failed[/red]"
            row = [host.hostname, status, host.project_path]
            if ros2_info:
                row.extend(["-", "-"])
            all_connected = False
            if verbose and analyzer.connection.last_error:
                console.print(f"[red]{host.hostname}:[/red] {analyzer.connection.last_error}")

        table.add_row(*row)

    console.print(table)

    if all_connected:
        console.print(f"\n[bold green]All {len(parsed_hosts)} hosts connected![/bold green]")
    else:
        console.print(f"\n[yellow]Some hosts failed to connect[/yellow]")
        sys.exit(1)


@main.command()
@click.argument("project_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--ssh", type=str, help="SSH host for live ROS2 system (user@host)")
@click.option("--output", "-o", type=click.Path(), help="Output JSON file path")
@click.option("--export-prometheus", type=click.Path(),
              help="Export Prometheus metrics to file")
@click.option("--check-http/--no-check-http", default=True,
              help="Check HTTP endpoints (default: enabled)")
@click.option("--check-systemd", is_flag=True, default=False,
              help="Check systemd service status (local or remote via --ssh)")
@click.option("--deployment-manifest", type=click.Path(exists=True),
              help="Path to deployment manifest YAML for systemd validation")
@click.option("--exclude", "-e", multiple=True,
              help="Glob patterns for paths to exclude")
@click.option("--use-default-excludes/--no-default-excludes", default=True,
              help="Apply default excludes for archive/backup/test dirs (default: enabled)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def validate(project_path: str, ssh: Optional[str], output: Optional[str],
             export_prometheus: Optional[str], check_http: bool,
             check_systemd: bool, deployment_manifest: Optional[str],
             exclude: tuple, use_default_excludes: bool, verbose: bool):
    """Validate static analysis against a live ROS2 system.

    Compares code analysis results against a running ROS2 system to find:
    - Topics in code but not active
    - Topics active but not in code
    - Type mismatches
    - Missing nodes
    - HTTP endpoint health (if --check-http)

    Requires a running ROS2 system (locally or via SSH).

    \b
    Examples:
        robomind validate ~/betaray
        robomind validate ~/betaray --ssh robot@nav.local
        robomind validate ~/betaray -o validation.json
        robomind validate ~/betaray --export-prometheus metrics.prom
    """
    from robomind.core.scanner import ProjectScanner
    from robomind.core.parser import PythonParser
    from robomind.ros2.node_extractor import ROS2NodeExtractor
    from robomind.ros2.topic_extractor import TopicExtractor
    from robomind.validators.live_validator import LiveValidator

    project_path = Path(project_path).resolve()
    exclude_patterns = list(exclude) if exclude else None

    console.print(f"\n[bold blue]RoboMind Live Validator[/bold blue]")
    console.print(f"Project: [cyan]{project_path}[/cyan]")
    if use_default_excludes:
        console.print(f"Default excludes: [green]enabled[/green]")
    if ssh:
        console.print(f"Remote:  [cyan]{ssh}[/cyan]")
    if check_http:
        console.print(f"HTTP checking: [green]enabled[/green]")
    console.print()

    try:
        # Extract ROS2 nodes from code
        console.print("[bold]Phase 1: Analyzing code...[/bold]")
        scanner = ProjectScanner(
            project_path,
            exclude_patterns=exclude_patterns,
            use_default_excludes=use_default_excludes,
        )
        scan_result = scanner.scan()

        parser = PythonParser()
        node_extractor = ROS2NodeExtractor()
        topic_extractor = TopicExtractor()
        all_nodes = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting...", total=None)
            for pf in scan_result.python_files:
                pr = parser.parse_file(pf.path)
                if pr and pr.has_ros2_imports():
                    nodes = node_extractor.extract_from_file(pf.path, pf.package_name)
                    all_nodes.extend(nodes)
                    topic_extractor.add_nodes(nodes)
            progress.update(task, completed=True)

        topic_graph = topic_extractor.build()
        console.print(f"  Code: {len(all_nodes)} nodes, {len(topic_graph.topics)} topics")

        # HTTP communication detection (for HTTP validation)
        http_comm_map = None
        if check_http:
            console.print("[bold]Phase 1.5: Detecting HTTP communication...[/bold]")
            from robomind.http import (
                HTTPEndpointExtractor,
                HTTPClientExtractor,
                build_communication_map,
            )

            endpoint_extractor = HTTPEndpointExtractor()
            client_extractor = HTTPClientExtractor()
            all_endpoints = []
            all_clients = []

            for pf in scan_result.python_files:
                all_endpoints.extend(endpoint_extractor.extract_from_file(pf.path))
                all_clients.extend(client_extractor.extract_from_file(pf.path))

            http_comm_map = build_communication_map(all_endpoints, all_clients, topic_graph)
            summary = http_comm_map.summary()
            console.print(f"  HTTP: {summary['http_endpoints']} endpoints, "
                         f"{summary['http_clients']} clients")

        # Validate against live system
        console.print("[bold]Phase 2: Validating against live system...[/bold]")
        validator = LiveValidator(
            all_nodes, topic_graph, ssh_host=ssh, http_comm_map=http_comm_map
        )
        result = validator.validate(check_http=check_http)

        if not result.validated:
            console.print(f"[bold yellow]ROS2 validation unavailable:[/bold yellow] {result.error}")
            if not (check_systemd or deployment_manifest):
                sys.exit(1)
            console.print("  [dim]Continuing with systemd checks...[/dim]")

        if result.live_info and result.live_info.available:
            console.print(f"  Live: {len(result.live_info.nodes)} nodes, "
                         f"{len(result.live_info.topics)} topics")

        # Phase 3: Systemd service validation
        systemd_results = []
        if check_systemd or deployment_manifest:
            console.print("[bold]Phase 3: Checking systemd services...[/bold]")
            from robomind.deployment.systemd_discovery import SystemdDiscovery

            discovery = SystemdDiscovery(ssh_host=ssh)

            if deployment_manifest:
                from robomind.deployment.manifest import load_deployment_manifest
                manifest = load_deployment_manifest(Path(deployment_manifest))

                # If SSH host is provided, find matching host in manifest
                if ssh:
                    # Check all hosts, match by SSH host or check local
                    for jetson_name, jetson_config in manifest.jetsons.items():
                        is_local = jetson_config.hostname in ("localhost", "thor.local")
                        is_ssh_target = ssh and (
                            jetson_config.hostname in ssh or
                            jetson_name in ssh
                        )
                        if is_local or is_ssh_target:
                            host_discovery = SystemdDiscovery(
                                ssh_host=ssh if is_ssh_target else None
                            )
                            services = host_discovery.discover_from_manifest(
                                jetson_config.systemd_services
                            )
                            for svc in services:
                                svc_info = {
                                    "host": jetson_name,
                                    "service": svc,
                                }
                                systemd_results.append(svc_info)
                else:
                    # Local only — check local host services
                    for jetson_name, jetson_config in manifest.jetsons.items():
                        if jetson_config.hostname in ("localhost", "thor.local"):
                            services = discovery.discover_from_manifest(
                                jetson_config.systemd_services
                            )
                            for svc in services:
                                systemd_results.append({
                                    "host": jetson_name,
                                    "service": svc,
                                })
            else:
                # Auto-discover with common patterns
                patterns = ["betaray*.service", "thor-*.service"]
                for pattern in patterns:
                    services = discovery.discover(pattern)
                    for svc in services:
                        systemd_results.append({
                            "host": "local" if not ssh else ssh,
                            "service": svc,
                        })

            # Display systemd results
            if systemd_results:
                systemd_table = Table(title="Systemd Services")
                systemd_table.add_column("Host", style="cyan")
                systemd_table.add_column("Service", style="white")
                systemd_table.add_column("Enabled", justify="center")
                systemd_table.add_column("Active", justify="center")
                systemd_table.add_column("Status", style="dim")

                for entry in systemd_results:
                    svc = entry["service"]
                    enabled_str = "[green]yes[/green]" if svc.is_enabled else "[red]no[/red]"
                    active_str = "[green]active[/green]" if svc.is_active else "[red]inactive[/red]"
                    systemd_table.add_row(
                        entry["host"],
                        svc.name,
                        enabled_str,
                        active_str,
                        svc.status,
                    )

                console.print(systemd_table)

                active_count = sum(1 for e in systemd_results if e["service"].is_active)
                total_count = len(systemd_results)
                if active_count == total_count:
                    console.print(f"  [green]All {total_count} services active[/green]")
                else:
                    console.print(
                        f"  [yellow]{active_count}/{total_count} services active[/yellow]"
                    )
            else:
                console.print("  [dim]No services found[/dim]")

        # Display results
        summary = result.summary()

        table = Table(title="Validation Results")
        table.add_column("Category", style="cyan")
        table.add_column("Count", justify="right")

        table.add_row("Total Differences", str(summary["total_diffs"]))
        table.add_row("Critical", str(summary["by_severity"]["critical"]))
        table.add_row("Errors", str(summary["by_severity"]["error"]))
        table.add_row("Warnings", str(summary["by_severity"]["warning"]))
        table.add_row("Info", str(summary["by_severity"]["info"]))

        console.print(table)

        # Show differences by type
        if result.diffs:
            console.print("\n[bold]Differences Found:[/bold]")

            # Group by type
            for severity in ["critical", "error", "warning"]:
                diffs = [d for d in result.diffs if d.severity.value == severity]
                if diffs:
                    color = {"critical": "red", "error": "yellow", "warning": "cyan"}[severity]
                    console.print(f"\n[{color}]{severity.upper()}:[/{color}]")
                    for diff in diffs[:10]:
                        console.print(f"  [{color}]\u2022[/{color}] {diff.message}")
                        if verbose and diff.recommendation:
                            console.print(f"    [dim]\u2192 {diff.recommendation}[/dim]")
                    if len(diffs) > 10:
                        console.print(f"    ... and {len(diffs) - 10} more")

        # Output to JSON
        if output:
            output_path = Path(output)
            output_data = result.to_dict()
            if systemd_results:
                output_data["systemd_services"] = [
                    {
                        "host": e["host"],
                        **e["service"].to_dict(),
                    }
                    for e in systemd_results
                ]
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            console.print(f"\n[green]Results saved to: {output_path}[/green]")

        # Export Prometheus metrics
        if export_prometheus:
            from robomind.validators.prometheus_exporter import export_prometheus_metrics

            prom_path = Path(export_prometheus)
            http_health_results = None
            if result.live_info and result.live_info.http_endpoints:
                http_health_results = {
                    ep: hr.to_dict()
                    for ep, hr in result.live_info.http_endpoints.items()
                }

            success = export_prometheus_metrics(
                output_path=prom_path,
                nodes=all_nodes,
                topic_graph=topic_graph,
                validation_result=result,
                http_comm_map=http_comm_map,
                http_health_results=http_health_results,
                project_name=project_path.name,
            )
            if success:
                console.print(f"[green]Prometheus metrics: {prom_path}[/green]")
            else:
                console.print(f"[yellow]Failed to export Prometheus metrics[/yellow]")

        if result.has_critical:
            console.print(f"\n[bold red]Critical issues found![/bold red]")
            sys.exit(2)
        elif result.has_errors:
            console.print(f"\n[bold yellow]Errors found, review recommended.[/bold yellow]")
        else:
            console.print(f"\n[bold green]Validation complete![/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument("project_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--output", "-o", type=click.Path(), default="ARCHITECTURE_REPORT.md",
              help="Output markdown file path")
@click.option("--exclude", "-e", multiple=True,
              help="Glob patterns for paths to exclude")
@click.option("--use-default-excludes/--no-default-excludes", default=True,
              help="Apply default excludes for archive/backup/test dirs (default: enabled)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def report(project_path: str, output: str, exclude: tuple,
           use_default_excludes: bool, verbose: bool):
    """Generate a comprehensive markdown report.

    Creates a detailed architecture report including:
    - Executive summary
    - Critical issues
    - Namespace analysis
    - Coupling hotspots
    - Node inventory
    - Recommendations

    \b
    Examples:
        robomind report ~/betaray
        robomind report ~/betaray -o my_report.md
    """
    from robomind.core.scanner import ProjectScanner
    from robomind.core.parser import PythonParser
    from robomind.ros2.node_extractor import ROS2NodeExtractor
    from robomind.ros2.topic_extractor import TopicExtractor
    from robomind.core.graph import build_system_graph
    from robomind.analyzers.coupling import analyze_coupling
    from robomind.reporters.markdown_reporter import generate_report

    project_path = Path(project_path).resolve()
    output_path = Path(output)
    exclude_patterns = list(exclude) if exclude else None

    console.print(f"\n[bold blue]RoboMind Report Generator[/bold blue]")
    console.print(f"Project: [cyan]{project_path}[/cyan]")
    console.print(f"Output:  [cyan]{output_path}[/cyan]")
    if use_default_excludes:
        console.print(f"Default excludes: [green]enabled[/green]")
    console.print()

    try:
        # Phase 1: Extract
        console.print("[bold]Phase 1: Analyzing project...[/bold]")
        scanner = ProjectScanner(
            project_path,
            exclude_patterns=exclude_patterns,
            use_default_excludes=use_default_excludes,
        )
        scan_result = scanner.scan()

        parser = PythonParser()
        node_extractor = ROS2NodeExtractor()
        topic_extractor = TopicExtractor()
        all_nodes = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting...", total=len(scan_result.python_files))
            for pf in scan_result.python_files:
                pr = parser.parse_file(pf.path)
                if pr and pr.has_ros2_imports():
                    nodes = node_extractor.extract_from_file(pf.path, pf.package_name)
                    all_nodes.extend(nodes)
                    topic_extractor.add_nodes(nodes)
                progress.advance(task)

        topic_graph = topic_extractor.build()
        console.print(f"  Found {len(all_nodes)} nodes, {len(topic_graph.topics)} topics")

        # Phase 2: Build graph and coupling
        console.print("[bold]Phase 2: Building graph...[/bold]")
        system_graph = build_system_graph(all_nodes, topic_graph)
        coupling = analyze_coupling(all_nodes, topic_graph)
        console.print(f"  Graph: {system_graph.stats()['total_nodes']} nodes, "
                     f"{system_graph.stats()['total_edges']} edges")

        # Phase 3: Generate report
        console.print("[bold]Phase 3: Generating report...[/bold]")
        result = generate_report(
            output_path=output_path,
            nodes=all_nodes,
            topic_graph=topic_graph,
            system_graph=system_graph,
            coupling=coupling,
            project_name=project_path.name,
            project_path=str(project_path),
        )

        if result.success:
            console.print(f"  [green]Report generated:[/green] {result.output_path}")
            console.print(f"  Size: {result.stats.get('file_size', 0):,} bytes")
            console.print(f"  Critical issues: {result.stats.get('critical_issues', 0)}")
            console.print(f"\n[bold green]Report complete![/bold green]")
        else:
            console.print(f"[bold red]Report failed:[/bold red] {result.error}")
            sys.exit(1)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument("project_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--from", "source", type=str, help="Source node name")
@click.option("--to", "target", type=str, help="Target node name")
@click.option("--topic", type=str, help="Trace all flows through a topic")
@click.option("--output", "-o", type=click.Path(), help="Output JSON file path")
@click.option("--exclude", "-e", multiple=True,
              help="Glob patterns for paths to exclude")
@click.option("--use-default-excludes/--no-default-excludes", default=True,
              help="Apply default excludes for archive/backup/test dirs (default: enabled)")
@click.option("--mermaid", is_flag=True, help="Output Mermaid diagram syntax")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def trace(project_path: str, source: Optional[str], target: Optional[str],
          topic: Optional[str], output: Optional[str], exclude: tuple,
          use_default_excludes: bool, mermaid: bool, verbose: bool):
    """Trace data flow paths through the system.

    Find all paths between two nodes, or trace all consumers of a topic.

    \b
    Examples:
        robomind trace ~/betaray --from sensor_node --to controller
        robomind trace ~/betaray --topic /cmd_vel
        robomind trace ~/betaray --from voice_node --to motor_node --mermaid
    """
    from robomind.core.scanner import ProjectScanner
    from robomind.core.parser import PythonParser
    from robomind.ros2.node_extractor import ROS2NodeExtractor
    from robomind.ros2.topic_extractor import TopicExtractor
    from robomind.analyzers.flow_tracer import FlowTracer

    project_path = Path(project_path).resolve()
    exclude_patterns = list(exclude) if exclude else None

    # Validate inputs
    if topic and (source or target):
        console.print("[bold red]Error:[/bold red] Use either --topic OR --from/--to, not both")
        sys.exit(1)

    if not topic and not (source and target):
        console.print("[bold red]Error:[/bold red] Specify --from and --to, or --topic")
        sys.exit(1)

    console.print(f"\n[bold blue]RoboMind Flow Tracer[/bold blue]")
    console.print(f"Project: [cyan]{project_path}[/cyan]")
    if topic:
        console.print(f"Tracing: [cyan]topic {topic}[/cyan]")
    else:
        console.print(f"Tracing: [cyan]{source}[/cyan] -> [cyan]{target}[/cyan]")
    console.print()

    try:
        # Extract nodes
        console.print("[bold]Phase 1: Analyzing project...[/bold]")
        scanner = ProjectScanner(
            project_path,
            exclude_patterns=exclude_patterns,
            use_default_excludes=use_default_excludes,
        )
        scan_result = scanner.scan()

        parser = PythonParser()
        node_extractor = ROS2NodeExtractor()
        topic_extractor = TopicExtractor()
        all_nodes = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting...", total=None)
            for pf in scan_result.python_files:
                pr = parser.parse_file(pf.path)
                if pr and pr.has_ros2_imports():
                    nodes = node_extractor.extract_from_file(pf.path, pf.package_name)
                    all_nodes.extend(nodes)
                    topic_extractor.add_nodes(nodes)
            progress.update(task, completed=True)

        topic_graph = topic_extractor.build()
        console.print(f"  Found {len(all_nodes)} nodes, {len(topic_graph.topics)} topics")

        # Trace
        console.print("[bold]Phase 2: Tracing flow...[/bold]")
        tracer = FlowTracer(all_nodes, topic_graph)

        if topic:
            result = tracer.trace_topic(topic)
        else:
            result = tracer.trace(source, target)

        if result.error:
            console.print(f"[bold red]Error:[/bold red] {result.error}")

            # Suggest similar nodes
            if "not found" in result.error:
                node_names = [n.name for n in all_nodes]
                search_term = source if "Source" in result.error else target
                matches = [n for n in node_names if search_term.lower() in n.lower()]
                if matches:
                    console.print("\n[yellow]Did you mean one of these?[/yellow]")
                    for m in matches[:10]:
                        console.print(f"  {m}")

            sys.exit(1)

        # Display results
        summary = result.summary()

        console.print(f"\n[green]Found {summary['paths_found']} paths[/green]")

        if result.bottlenecks:
            console.print(f"\n[yellow]Bottlenecks (single points of failure):[/yellow]")
            for bn in result.bottlenecks:
                console.print(f"  [yellow]\u2022[/yellow] {bn}")

        # Show paths
        console.print(f"\n[bold]Paths:[/bold]")
        for i, path in enumerate(result.paths[:10], 1):
            console.print(f"\n  [cyan]Path {i}[/cyan] (length {path.length}):")
            console.print(f"    {' -> '.join(path.nodes)}")
            if verbose and path.topics:
                console.print(f"    Topics: {' -> '.join(path.topics)}")

        if len(result.paths) > 10:
            console.print(f"\n  ... and {len(result.paths) - 10} more paths")

        # Mermaid output
        if mermaid and result.paths:
            console.print(f"\n[bold]Mermaid Diagram:[/bold]")
            console.print("```mermaid")
            console.print(result.paths[0].to_mermaid())
            console.print("```")

        # Flow summary
        if verbose:
            flow_summary = tracer.get_flow_summary()
            console.print(f"\n[bold]System Flow Summary:[/bold]")
            console.print(f"  Entry points: {', '.join(flow_summary['entry_points'][:5])}")
            console.print(f"  Exit points: {', '.join(flow_summary['exit_points'][:5])}")
            console.print(f"  High traffic nodes:")
            for item in flow_summary['high_traffic_nodes'][:5]:
                console.print(f"    {item['node']}: {item['connections']} connections")

        # Output to JSON
        if output:
            output_path = Path(output)
            with open(output_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            console.print(f"\n[green]Results saved to: {output_path}[/green]")

        console.print(f"\n[bold green]Trace complete![/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument("project_path", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output file for recommendations (JSON)")
@click.option("--trace-launch", type=click.Path(exists=True), help="Filter to nodes deployed by this launch file")
@click.option("--min-severity", type=click.Choice(["critical", "high", "medium", "low"]),
              default="low", help="Minimum severity to report")
@click.option("-v", "--verbose", is_flag=True, help="Show detailed output")
def recommend(project_path: str, output: str, trace_launch: str, min_severity: str, verbose: bool):
    """
    Generate actionable recommendations for improving ROS2 code.

    Analyzes the codebase for issues and suggests specific fixes:

    \b
    - CRITICAL: Safety issues (e.g., emergency stop inconsistencies)
    - HIGH: Bugs that will cause problems (e.g., relative topic names)
    - MEDIUM: Code quality issues (e.g., orphaned subscribers)
    - LOW: Style/consistency suggestions

    Examples:

    \b
      robomind recommend ~/my_robot
      robomind recommend ~/project --min-severity high
      robomind recommend ~/project --trace-launch launch/robot.launch.py
      robomind recommend ~/project -o recommendations.json
    """
    import json
    from pathlib import Path
    from robomind.core.scanner import ProjectScanner
    from robomind.ros2.node_extractor import ROS2NodeExtractor
    from robomind.ros2.topic_extractor import TopicExtractor
    from robomind.analyzers.recommendations import generate_recommendations, Severity

    project_path = Path(project_path).resolve()

    console.print(f"\n[bold blue]RoboMind Recommendations[/bold blue]")
    console.print(f"Project: {project_path}")
    if trace_launch:
        console.print(f"Launch filter: {trace_launch}")
    console.print()

    try:
        # Phase 1: Scan
        console.print("[bold]Phase 1: Scanning project...[/bold]")
        scanner = ProjectScanner(project_path)
        scan = scanner.scan()
        console.print(f"  Found {len(scan.python_files)} Python files")

        # Find ROS2 files
        ros2_files = []
        for pf in scan.python_files:
            try:
                with open(pf.path, 'r') as f:
                    content = f.read()
                if 'rclpy' in content or 'from rclpy' in content:
                    ros2_files.append(pf.path)
            except Exception:
                pass
        console.print(f"  Found {len(ros2_files)} ROS2 files")

        # Phase 2: Extract nodes
        console.print("[bold]Phase 2: Extracting ROS2 nodes...[/bold]")
        extractor = ROS2NodeExtractor()
        nodes = []
        for f in ros2_files:
            nodes.extend(extractor.extract_from_file(f))
        console.print(f"  Found {len(nodes)} ROS2 nodes")

        # Optional: Filter by launch file
        if trace_launch:
            from robomind.deployment import trace_launch_file

            console.print("[bold]Phase 2.5: Filtering to deployed nodes...[/bold]")
            trace = trace_launch_file(Path(trace_launch), project_root=project_path)
            traced_names = set()
            for node in trace.nodes:
                traced_names.add(node.name.lower().replace("_", "").replace("-", ""))
                traced_names.add(node.executable.lower().replace("_", "").replace("-", ""))

            def is_deployed(node):
                node_names = [
                    node.name.lower().replace("_", "").replace("-", ""),
                    (node.class_name or "").lower().replace("_", "").replace("-", ""),
                    Path(node.file_path).stem.lower().replace("_", "").replace("-", "") if node.file_path else "",
                ]
                for traced in traced_names:
                    for node_name in node_names:
                        if node_name and (traced in node_name or node_name in traced):
                            return True
                return False

            original_count = len(nodes)
            nodes = [n for n in nodes if is_deployed(n)]
            console.print(f"  Filtered to {len(nodes)} deployed nodes (from {original_count})")

        # Phase 3: Build topic graph
        console.print("[bold]Phase 3: Building topic graph...[/bold]")
        topic_ext = TopicExtractor()
        topic_ext.add_nodes(nodes)
        topic_graph = topic_ext.build()
        console.print(f"  Found {len(topic_graph.topics)} topics")

        # Phase 4: Generate recommendations
        console.print("[bold]Phase 4: Analyzing for issues...[/bold]")
        report = generate_recommendations(nodes, topic_graph)

        # Filter by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        min_level = severity_order[min_severity]
        filtered = [r for r in report.recommendations
                   if severity_order[r.severity.value] <= min_level]

        console.print(f"  Found {len(filtered)} issues (severity >= {min_severity})")
        console.print()

        # Display recommendations
        console.print("[bold]=== RECOMMENDATIONS ===[/bold]\n")

        severity_colors = {
            "critical": "red bold",
            "high": "yellow",
            "medium": "cyan",
            "low": "dim",
        }

        for i, rec in enumerate(filtered, 1):
            color = severity_colors[rec.severity.value]
            console.print(f"[{color}]{i}. [{rec.severity.value.upper()}] {rec.title}[/{color}]")
            console.print(f"   [dim]Category:[/dim] {rec.category.value}")
            console.print(f"   [dim]Impact:[/dim] {rec.impact}")

            if rec.affected_nodes and verbose:
                console.print(f"   [dim]Nodes:[/dim] {', '.join(rec.affected_nodes[:5])}")

            if rec.fixes:
                console.print(f"   [green]Fixes available: {len(rec.fixes)}[/green]")
                for fix in rec.fixes[:3]:
                    rel_path = Path(fix.file_path).name
                    console.print(f"     [dim]{rel_path}:{fix.line_number}[/dim]")
                    console.print(f"       {fix.original} [green]->[/green] {fix.replacement}")
                if len(rec.fixes) > 3:
                    console.print(f"     [dim]... and {len(rec.fixes) - 3} more fixes[/dim]")
            console.print()

        # Summary
        by_severity = {}
        for rec in filtered:
            s = rec.severity.value
            by_severity[s] = by_severity.get(s, 0) + 1

        console.print("[bold]=== SUMMARY ===[/bold]")
        console.print(f"Total issues: {len(filtered)}")
        for sev in ["critical", "high", "medium", "low"]:
            if sev in by_severity:
                console.print(f"  {sev.upper()}: {by_severity[sev]}")
        fixable = sum(1 for r in filtered if r.fixes)
        console.print(f"Auto-fixable: {fixable}")

        # Save to file
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
            console.print(f"\n[green]Saved to: {output_path}[/green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command(name='deep-analyze')
@click.argument("project_path", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output file for report (JSON)")
@click.option("--trace-launch", type=click.Path(exists=True), help="Filter to deployed nodes")
@click.option("--min-severity", type=click.Choice(["critical", "high", "medium", "low"]),
              default="low", help="Minimum severity to report")
@click.option("--enable", multiple=True,
              type=click.Choice(["qos", "timing", "security", "architecture", "complexity", "message", "parameter"]),
              help="Enable specific analyzers (default: all)")
@click.option("-v", "--verbose", is_flag=True, help="Show detailed output")
def deep_analyze_cmd(project_path: str, output: str, trace_launch: str, min_severity: str,
                     enable: tuple, verbose: bool):
    """
    Run comprehensive deep analysis on ROS2 codebase.

    Combines multiple analyzers for thorough code review:

    \b
    - QoS: Quality of Service compatibility between pub/sub
    - Timing: Callback chains, blocking operations, timer frequencies
    - Security: Hardcoded secrets, injection vulnerabilities
    - Architecture: Circular dependencies, dead code, anti-patterns
    - Complexity: Cyclomatic complexity, nesting depth
    - Message: Type mismatches, deprecated types
    - Parameter: Missing validation, naming conventions

    Examples:

    \b
      robomind deep-analyze ~/my_robot
      robomind deep-analyze ~/project --min-severity high
      robomind deep-analyze ~/project --enable security --enable timing
      robomind deep-analyze ~/project --trace-launch launch/robot.launch.py
    """
    import json
    from pathlib import Path
    from robomind.core.scanner import ProjectScanner
    from robomind.ros2.node_extractor import ROS2NodeExtractor
    from robomind.ros2.topic_extractor import TopicExtractor
    from robomind.analyzers.deep_analyzer import DeepAnalyzer

    project_path = Path(project_path).resolve()

    console.print(f"\n[bold blue]RoboMind Deep Analysis[/bold blue]")
    console.print(f"Project: {project_path}")
    if trace_launch:
        console.print(f"Launch filter: {trace_launch}")
    console.print()

    try:
        # Determine which analyzers to enable
        all_analyzers = {"qos", "timing", "security", "architecture", "complexity", "message", "parameter"}
        enabled = set(enable) if enable else all_analyzers

        console.print(f"[dim]Enabled analyzers: {', '.join(sorted(enabled))}[/dim]\n")

        # Phase 1: Scan
        console.print("[bold]Phase 1: Scanning project...[/bold]")
        scanner = ProjectScanner(project_path)
        scan = scanner.scan()

        # Find ROS2 files
        ros2_files = []
        for pf in scan.python_files:
            try:
                with open(pf.path, 'r') as f:
                    content = f.read()
                if 'rclpy' in content or 'from rclpy' in content:
                    ros2_files.append(pf.path)
            except Exception:
                pass
        console.print(f"  Found {len(ros2_files)} ROS2 files")

        # Phase 2: Extract nodes
        console.print("[bold]Phase 2: Extracting ROS2 nodes...[/bold]")
        extractor = ROS2NodeExtractor()
        nodes = []
        for f in ros2_files:
            nodes.extend(extractor.extract_from_file(f))
        console.print(f"  Found {len(nodes)} ROS2 nodes")

        # Optional: Filter by launch file
        launched_nodes = set()
        if trace_launch:
            from robomind.deployment import trace_launch_file

            console.print("[bold]Phase 2.5: Filtering to deployed nodes...[/bold]")
            trace = trace_launch_file(Path(trace_launch), project_root=project_path)

            for node in trace.nodes:
                launched_nodes.add(node.name.lower().replace("_", "").replace("-", ""))
                launched_nodes.add(node.executable.lower().replace("_", "").replace("-", ""))

            def is_deployed(node):
                node_names = [
                    node.name.lower().replace("_", "").replace("-", ""),
                    (node.class_name or "").lower().replace("_", "").replace("-", ""),
                    Path(node.file_path).stem.lower().replace("_", "").replace("-", "") if node.file_path else "",
                ]
                for traced in launched_nodes:
                    for node_name in node_names:
                        if node_name and (traced in node_name or node_name in traced):
                            return True
                return False

            original_count = len(nodes)
            nodes = [n for n in nodes if is_deployed(n)]
            console.print(f"  Filtered to {len(nodes)} deployed nodes (from {original_count})")

        # Phase 3: Build topic graph
        console.print("[bold]Phase 3: Building topic graph...[/bold]")
        topic_ext = TopicExtractor()
        topic_ext.add_nodes(nodes)
        topic_graph = topic_ext.build()
        console.print(f"  Found {len(topic_graph.topics)} topics")

        # Phase 4: Deep Analysis
        console.print("[bold]Phase 4: Running deep analysis...[/bold]")
        analyzer = DeepAnalyzer()
        analyzer.add_nodes(nodes)
        analyzer.add_topic_graph(topic_graph)
        if launched_nodes:
            analyzer.set_launched_nodes(launched_nodes)

        report = analyzer.analyze(
            enable_qos="qos" in enabled,
            enable_timing="timing" in enabled,
            enable_security="security" in enabled,
            enable_architecture="architecture" in enabled,
            enable_complexity="complexity" in enabled,
            enable_message="message" in enabled,
            enable_parameter="parameter" in enabled,
        )

        # Filter by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        min_level = severity_order[min_severity]
        filtered = [f for f in report.all_findings if severity_order.get(f.severity, 4) <= min_level]

        console.print(f"  Found {len(filtered)} issues (severity >= {min_severity})")
        console.print()

        # Display findings by category
        console.print("[bold]=== DEEP ANALYSIS RESULTS ===[/bold]\n")

        severity_colors = {
            "critical": "red bold",
            "high": "yellow",
            "medium": "cyan",
            "low": "dim",
        }

        # Group by category
        categories = {
            "security": ("Security Vulnerabilities", []),
            "architecture": ("Architecture Issues", []),
            "timing": ("Timing Issues", []),
            "qos": ("QoS Compatibility", []),
            "complexity": ("Complexity Issues", []),
            "message": ("Message Type Issues", []),
            "parameter": ("Parameter Issues", []),
        }

        for finding in filtered:
            if finding.category in categories:
                categories[finding.category][1].append(finding)

        for cat_key, (cat_name, cat_findings) in categories.items():
            if not cat_findings:
                continue

            console.print(f"[bold]{cat_name}[/bold] ({len(cat_findings)})")
            for i, f in enumerate(cat_findings[:10], 1):
                color = severity_colors.get(f.severity, "white")
                console.print(f"  [{color}]{i}. [{f.severity.upper()}] {f.title}[/{color}]")
                if verbose:
                    console.print(f"     {f.description[:80]}...")
                    if f.file_path:
                        console.print(f"     [dim]{Path(f.file_path).name}:{f.line_number or '?'}[/dim]")
                    if f.recommendation:
                        console.print(f"     [green]Fix:[/green] {f.recommendation[:60]}...")
            if len(cat_findings) > 10:
                console.print(f"  [dim]... and {len(cat_findings) - 10} more[/dim]")
            console.print()

        # Timing chains
        if report.callback_chains and "timing" in enabled:
            console.print("[bold]Callback Chains Detected[/bold]")
            console.print(f"  Total chains: {len(report.callback_chains)}")
            if report.critical_path:
                console.print(f"  Critical path: {' → '.join(report.critical_path.nodes[:5])}")
                console.print(f"  Hops: {report.critical_path.total_hops}")
            console.print()

        # Summary
        console.print("[bold]=== SUMMARY ===[/bold]")
        console.print(f"Total findings: {len(filtered)}")
        console.print(f"  [red]CRITICAL: {report.critical_count}[/red]")
        console.print(f"  [yellow]HIGH: {report.high_count}[/yellow]")
        console.print(f"  [cyan]MEDIUM: {report.medium_count}[/cyan]")
        console.print(f"  [dim]LOW: {report.low_count}[/dim]")

        # Save to file
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
            console.print(f"\n[green]Report saved to: {output_path}[/green]")

        console.print("\n[bold green]Deep analysis complete![/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument("project_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--topic", type=str, help="Analyze impact of topic change/removal")
@click.option("--node", type=str, help="Analyze impact of node removal")
@click.option("--file", "file_path", type=str, help="Analyze impact of file change")
@click.option("--message-type", type=str, help="Analyze impact of message type change")
@click.option("--output", "-o", type=click.Path(), help="Output JSON file path")
@click.option("--exclude", "-e", multiple=True,
              help="Glob patterns for paths to exclude")
@click.option("--use-default-excludes/--no-default-excludes", default=True,
              help="Apply default excludes for archive/backup/test dirs (default: enabled)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def impact(project_path: str, topic: Optional[str], node: Optional[str],
           file_path: Optional[str], message_type: Optional[str],
           output: Optional[str], exclude: tuple, use_default_excludes: bool,
           verbose: bool):
    """Analyze impact of changes to the ROS2 system.

    Shows what breaks if you rename a topic, remove a node, change a message
    type, or modify a source file.

    \b
    Examples:
        robomind impact ~/betaray --topic /cmd_vel
        robomind impact ~/betaray --node motor_controller_node
        robomind impact ~/betaray --file path/to/file.py
        robomind impact ~/betaray --message-type sensor_msgs/msg/LaserScan
    """
    from robomind.core.scanner import ProjectScanner
    from robomind.core.parser import PythonParser
    from robomind.ros2.node_extractor import ROS2NodeExtractor
    from robomind.ros2.topic_extractor import TopicExtractor
    from robomind.analyzers.impact_analyzer import ImpactAnalyzer

    project_path = Path(project_path).resolve()
    exclude_patterns = list(exclude) if exclude else None

    # Validate: exactly one target must be specified
    targets = [t for t in [topic, node, file_path, message_type] if t]
    if len(targets) != 1:
        console.print("[bold red]Error:[/bold red] Specify exactly one of --topic, --node, --file, or --message-type")
        sys.exit(1)

    # Determine target
    if topic:
        target, target_type, target_label = topic, "topic", f"topic {topic}"
    elif node:
        target, target_type, target_label = node, "node", f"node {node}"
    elif file_path:
        target, target_type, target_label = file_path, "file", f"file {file_path}"
    else:
        target, target_type, target_label = message_type, "message_type", f"message type {message_type}"

    console.print(f"\n[bold blue]RoboMind Impact Analysis[/bold blue]")
    console.print(f"Project: [cyan]{project_path}[/cyan]")
    console.print(f"Target: [cyan]{target_label}[/cyan]")
    console.print()

    try:
        # Phase 1: Extract nodes
        console.print("[bold]Phase 1: Analyzing project...[/bold]")
        scanner = ProjectScanner(
            project_path,
            exclude_patterns=exclude_patterns,
            use_default_excludes=use_default_excludes,
        )
        scan_result = scanner.scan()

        parser = PythonParser()
        node_extractor = ROS2NodeExtractor()
        topic_extractor = TopicExtractor()
        all_nodes = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting...", total=None)
            for pf in scan_result.python_files:
                pr = parser.parse_file(pf.path)
                if pr and pr.has_ros2_imports():
                    nodes = node_extractor.extract_from_file(pf.path, pf.package_name)
                    all_nodes.extend(nodes)
                    topic_extractor.add_nodes(nodes)
            progress.update(task, completed=True)

        topic_graph = topic_extractor.build()
        console.print(f"  Found {len(all_nodes)} nodes, {len(topic_graph.topics)} topics")

        # Phase 2: Impact analysis
        console.print("[bold]Phase 2: Analyzing impact...[/bold]")
        analyzer = ImpactAnalyzer(all_nodes, topic_graph)

        if target_type == "topic":
            result = analyzer.analyze_topic_change(target)
        elif target_type == "node":
            result = analyzer.analyze_node_removal(target)
        elif target_type == "message_type":
            result = analyzer.analyze_message_type_change(target)
        else:
            result = analyzer.analyze_file_change(target)

        summary = result.summary()

        if summary["total_affected"] == 0:
            console.print(f"\n[yellow]No affected entities found for {target_label}[/yellow]")

            # Suggest matches
            if target_type == "topic":
                all_topics = list(analyzer._topic_publishers.keys()) + list(analyzer._topic_subscribers.keys())
                matches = [t for t in set(all_topics) if target.lower() in t.lower()]
                if matches:
                    console.print("\n[yellow]Did you mean one of these topics?[/yellow]")
                    for m in sorted(matches)[:10]:
                        console.print(f"  {m}")
            elif target_type == "node":
                matches = [n.name for n in all_nodes if target.lower() in n.name.lower()]
                if matches:
                    console.print("\n[yellow]Did you mean one of these nodes?[/yellow]")
                    for m in matches[:10]:
                        console.print(f"  {m}")
        else:
            # Display results
            severity_colors = {
                "critical": "red bold",
                "high": "yellow",
                "medium": "cyan",
                "low": "dim",
            }

            console.print(f"\n[bold]=== IMPACT ANALYSIS ===[/bold]")
            console.print(f"  Total affected: {summary['total_affected']}")
            console.print(f"  Directly affected: {summary['directly_affected']}")
            console.print(f"  Cascade affected: {summary['cascade_affected']}")
            for sev, count in sorted(summary["by_severity"].items(),
                                     key=lambda x: ["critical", "high", "medium", "low"].index(x[0])):
                color = severity_colors.get(sev, "white")
                console.print(f"  [{color}]{sev.upper()}: {count}[/{color}]")

            if result.directly_affected:
                console.print(f"\n[bold]Directly Affected:[/bold]")
                for item in result.directly_affected:
                    color = severity_colors.get(item.severity, "white")
                    console.print(f"  [{color}][{item.severity.upper()}][/{color}] {item.name} ({item.kind})")
                    console.print(f"    {item.description}")
                    if verbose and item.file_path:
                        console.print(f"    [dim]{item.file_path}[/dim]")

            if result.cascade_affected:
                console.print(f"\n[bold]Cascade Affected:[/bold]")
                for item in result.cascade_affected:
                    color = severity_colors.get(item.severity, "white")
                    console.print(f"  [{color}][{item.severity.upper()}][/{color}] {item.name} ({item.kind})")
                    console.print(f"    {item.description}")
                    if verbose and item.file_path:
                        console.print(f"    [dim]{item.file_path}[/dim]")

        # Output to JSON
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            console.print(f"\n[green]Results saved to: {output_path}[/green]")

        console.print(f"\n[bold green]Impact analysis complete![/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
def info():
    """Show information about RoboMind."""
    from robomind import __version__

    console.print(f"\n[bold blue]RoboMind v{__version__}[/bold blue]")
    console.print("Rapid Prototyping System for Autonomous ROS2 Robots\n")

    console.print("[bold]Capabilities:[/bold]")
    console.print("  [green]\u2713[/green] Scan project directories for Python/ROS2 files")
    console.print("  [green]\u2713[/green] Parse Python AST for classes, functions, imports")
    console.print("  [green]\u2713[/green] Extract ROS2 nodes (publishers, subscribers, timers)")
    console.print("  [green]\u2713[/green] Extract ROS2 services, actions, parameters")
    console.print("  [green]\u2713[/green] Build topic connection graph")
    console.print("  [green]\u2713[/green] Parse launch files for topology")
    console.print("  [green]\u2713[/green] Extract YAML parameter configurations")
    console.print("  [green]\u2713[/green] Build NetworkX dependency graph")
    console.print("  [green]\u2713[/green] Analyze component coupling strength")
    console.print("  [green]\u2713[/green] Export JSON system graph")
    console.print("  [green]\u2713[/green] Export YAML AI context (token-efficient)")
    console.print("  [green]\u2713[/green] Generate HTML visualization (D3.js interactive)")
    console.print("  [green]\u2713[/green] SSH remote analysis (distributed systems)")
    console.print("  [green]\u2713[/green] Validate against live ROS2 system")
    console.print("  [green]\u2713[/green] Generate markdown architecture reports")
    console.print("  [green]\u2713[/green] Trace data flow paths between nodes")
    console.print("  [green]\u2713[/green] Default excludes (archive/backup/test/deprecated)")
    console.print("  [green]\u2713[/green] Confidence scoring (filter false positives)")
    console.print("  [green]\u2713[/green] Impact analysis (what breaks if X changes?)")
    console.print("  [green]\u2713[/green] Message definition parsing (.msg/.srv/.action)")

    console.print("\n[bold]Quick Start:[/bold]")
    console.print("  robomind scan ~/my_robot_project")
    console.print("  robomind analyze ~/my_robot_project -o ./analysis/")
    console.print("  robomind analyze ~/project --min-confidence 0.5  # filter low-confidence")
    console.print("  robomind launch ~/my_robot_project -v")
    console.print("  robomind graph ~/my_robot_project --coupling -v")
    console.print("  robomind visualize ~/my_robot_project -o viz.html --open")
    console.print("  robomind validate ~/my_robot_project")
    console.print("  robomind report ~/my_robot_project -o REPORT.md")
    console.print("  robomind trace ~/project --from sensor --to controller")
    console.print("  robomind impact ~/project --topic /cmd_vel")
    console.print("  robomind remote robot@jetson.local --ros2-info")

    console.print("\n[bold]Distributed Analysis:[/bold]")
    console.print("  robomind analyze ~/local_project \\")
    console.print("      --remote robot@nav.local:~/project \\")
    console.print("      --remote robot@ai.local:~/project")
    console.print()


if __name__ == "__main__":
    main()
