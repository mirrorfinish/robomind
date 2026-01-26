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
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def scan(project_path: str, output: Optional[str], verbose: bool):
    """Scan a project directory for Python files and ROS2 packages.

    PROJECT_PATH: Path to the robotics project to scan
    """
    from robomind.core.scanner import ProjectScanner

    project_path = Path(project_path).resolve()

    console.print(f"\n[bold blue]RoboMind Scanner[/bold blue]")
    console.print(f"Scanning: [cyan]{project_path}[/cyan]\n")

    try:
        scanner = ProjectScanner(project_path)

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
              type=click.Choice(["json", "yaml", "html"]),
              default=["json", "yaml", "html"],
              help="Output formats (can specify multiple)")
@click.option("--remote", "-r", multiple=True,
              help="Remote hosts to analyze (user@host:path)")
@click.option("--key", "-k", type=click.Path(exists=True),
              help="SSH private key file for remote connections")
@click.option("--keep-remote", is_flag=True,
              help="Keep local copies of remote code after analysis")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def analyze(project_path: str, output: str, formats: tuple, remote: tuple,
            key: Optional[str], keep_remote: bool, verbose: bool):
    """Perform full analysis of a ROS2 project.

    Extracts nodes, topics, parameters, and generates structured output.

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

    console.print(f"\n[bold blue]RoboMind Analyzer[/bold blue]")
    console.print(f"Project: [cyan]{project_path}[/cyan]")
    console.print(f"Output:  [cyan]{output_dir}[/cyan]")
    console.print(f"Formats: [cyan]{', '.join(formats)}[/cyan]")

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
        scanner = ProjectScanner(project_path)
        scan_result = scanner.scan()
        console.print(f"  Found {len(scan_result.python_files)} Python files, "
                     f"{len(scan_result.packages)} packages")

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

        # Statistics
        total_publishers = sum(len(n.publishers) for n in all_nodes)
        total_subscribers = sum(len(n.subscribers) for n in all_nodes)
        total_timers = sum(len(n.timers) for n in all_nodes)
        total_params = sum(len(n.parameters) for n in all_nodes)

        console.print(f"  Found {len(all_nodes)} ROS2 nodes")
        console.print(f"  Publishers: {total_publishers}, Subscribers: {total_subscribers}")
        console.print(f"  Timers: {total_timers}, Parameters: {total_params}")
        console.print(f"  Topics: {len(topic_graph.topics)} "
                     f"({len(topic_graph.get_connected_topics())} connected)")

        # Phase 4: Build graph and coupling (for exports)
        console.print("[bold]Phase 4: Building system graph...[/bold]")
        from robomind.core.graph import build_system_graph
        from robomind.analyzers.coupling import analyze_coupling

        system_graph = build_system_graph(all_nodes, topic_graph)
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
            )

            if result.success:
                console.print(f"  [green]HTML:[/green] {html_path}")
            else:
                console.print(f"  [red]HTML failed:[/red] {result.error}")

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
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def launch(project_path: str, output: Optional[str], verbose: bool):
    """Analyze ROS2 launch files and parameter configs.

    Extracts launch topology, node sequences, and parameter configurations.

    PROJECT_PATH: Path to the robotics project to analyze
    """
    from robomind.core.scanner import ProjectScanner
    from robomind.ros2.launch_analyzer import LaunchFileAnalyzer, LaunchTopology
    from robomind.ros2.param_extractor import ConfigScanner

    project_path = Path(project_path).resolve()

    console.print(f"\n[bold blue]RoboMind Launch Analyzer[/bold blue]")
    console.print(f"Project: [cyan]{project_path}[/cyan]\n")

    try:
        # Phase 1: Find launch files
        console.print("[bold]Phase 1: Scanning for launch files...[/bold]")
        scanner = ProjectScanner(project_path)
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
@click.option("--coupling/--no-coupling", default=True, help="Include coupling analysis")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def graph(project_path: str, output: Optional[str], graphml: Optional[str],
          coupling: bool, verbose: bool):
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

    console.print(f"\n[bold blue]RoboMind Graph Builder[/bold blue]")
    console.print(f"Project: [cyan]{project_path}[/cyan]\n")

    try:
        # Phase 1: Scan and extract
        console.print("[bold]Phase 1: Extracting ROS2 nodes...[/bold]")
        scanner = ProjectScanner(project_path)
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
@click.option("--open", "open_browser", is_flag=True, help="Open in browser after generation")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def visualize(project_path: str, output: str, open_browser: bool, verbose: bool):
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

    console.print(f"\n[bold blue]RoboMind Visualizer[/bold blue]")
    console.print(f"Project: [cyan]{project_path}[/cyan]")
    console.print(f"Output:  [cyan]{output_path}[/cyan]\n")

    try:
        # Phase 1: Extract ROS2 nodes
        console.print("[bold]Phase 1: Extracting ROS2 nodes...[/bold]")
        scanner = ProjectScanner(project_path)
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

    console.print("\n[bold]Test Coverage:[/bold] 192 tests passing")

    console.print("\n[bold]Quick Start:[/bold]")
    console.print("  robomind scan ~/my_robot_project")
    console.print("  robomind analyze ~/my_robot_project -o ./analysis/")
    console.print("  robomind launch ~/my_robot_project -v")
    console.print("  robomind graph ~/my_robot_project --coupling -v")
    console.print("  robomind visualize ~/my_robot_project -o viz.html --open")
    console.print("  robomind remote robot@jetson.local --ros2-info")

    console.print("\n[bold]Distributed Analysis:[/bold]")
    console.print("  robomind analyze ~/local_project \\")
    console.print("      --remote robot@nav.local:~/project \\")
    console.print("      --remote robot@ai.local:~/project")
    console.print()


if __name__ == "__main__":
    main()
