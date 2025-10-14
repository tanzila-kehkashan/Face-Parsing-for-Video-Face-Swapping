class Job {
  final String id;
  final String status;
  final String? outputPath;
  final String? createdAt;

  Job({
    required this.id,
    required this.status,
    this.outputPath,
    this.createdAt,
  });

  factory Job.fromJson(Map<String, dynamic> json) {
    return Job(
      id: json['id'] ?? '',
      status: json['status'] ?? 'unknown',
      outputPath: json['output_path'],
      createdAt: json['created_at'],
    );
  }
}