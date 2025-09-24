import 'package:flutter/material.dart';

import '../../theme/app_colors.dart';

class FoldersPage extends StatelessWidget {
  const FoldersPage({super.key});

  static final _folders = [
    const _Folder('Personal', 12, Color(0xFF306BFF)),
    const _Folder('Work', 5, Color(0xFF22C55E)),
    const _Folder('Projects', 3, Color(0xFF8B5CF6)),
    const _Folder('Ideas', 8, Color(0xFFF97316)),
    const _Folder('Travel', 2, Color(0xFF14B8A6)),
    const _Folder('Recipes', 10, Color(0xFFEF4444)),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(24, 16, 24, 8),
              child: Row(
                children: [
                  const Text(
                    'Folders',
                    style: TextStyle(
                      fontSize: 28,
                      fontWeight: FontWeight.w700,
                      letterSpacing: -0.3,
                    ),
                  ),
                  const Spacer(),
                  IconButton(
                    onPressed: () {},
                    icon: Container(
                      width: 40,
                      height: 40,
                      decoration: const BoxDecoration(
                        shape: BoxShape.circle,
                        color: AppColors.primaryBlue,
                      ),
                      child: const Icon(Icons.add, color: Colors.white),
                    ),
                  ),
                ],
              ),
            ),
            Expanded(
              child: ListView.builder(
                padding: const EdgeInsets.symmetric(horizontal: 24),
                itemCount: _folders.length,
                itemBuilder: (context, index) {
                  final folder = _folders[index];
                  return Padding(
                    padding: const EdgeInsets.only(bottom: 12),
                    child: ListTile(
                      onTap: () {},
                      tileColor: Colors.white,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(18),
                      ),
                      leading: Container(
                        width: 48,
                        height: 48,
                        decoration: BoxDecoration(
                          color: folder.color.withOpacity(0.15),
                          borderRadius: BorderRadius.circular(16),
                        ),
                        child: Icon(Icons.folder, color: folder.color),
                      ),
                      title: Text(
                        folder.name,
                        style: const TextStyle(fontWeight: FontWeight.w700),
                      ),
                      subtitle: Text(
                        '${folder.noteCount} notes',
                        style: const TextStyle(color: AppColors.textSecondary),
                      ),
                      trailing: const Icon(Icons.more_horiz),
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _Folder {
  const _Folder(this.name, this.noteCount, this.color);

  final String name;
  final int noteCount;
  final Color color;
}
