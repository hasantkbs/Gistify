import 'package:flutter/material.dart';

import '../../theme/app_colors.dart';

class FolderDetailPage extends StatelessWidget {
  const FolderDetailPage({
    super.key,
    required this.folderName,
    required this.noteCount,
  });

  final String folderName;
  final int noteCount;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(folderName), centerTitle: false),
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              '$noteCount notes',
              style: Theme.of(
                context,
              ).textTheme.titleMedium?.copyWith(color: AppColors.textSecondary),
            ),
            const SizedBox(height: 24),
            Expanded(
              child: Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: const [
                    Icon(
                      Icons.folder_copy_outlined,
                      size: 48,
                      color: AppColors.textSecondary,
                    ),
                    SizedBox(height: 12),
                    Text(
                      'Folder is empty',
                      style: TextStyle(
                        fontWeight: FontWeight.w600,
                        color: AppColors.textSecondary,
                      ),
                    ),
                    SizedBox(height: 6),
                    Text(
                      'Upload notes to see them here.',
                      style: TextStyle(color: AppColors.textSecondary),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
