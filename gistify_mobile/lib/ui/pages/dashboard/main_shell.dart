import 'package:flutter/material.dart';

import '../../theme/app_colors.dart';
import '../folders/folders_page.dart';
import '../notes/notes_page.dart';
import '../settings/settings_page.dart';
import '../upload/upload_page.dart';

class MainShell extends StatefulWidget {
  const MainShell({super.key});

  static const route = '/home';

  @override
  State<MainShell> createState() => _MainShellState();
}

class _MainShellState extends State<MainShell> {
  int _currentIndex = 0;

  final _pages = const [
    UploadPage(),
    NotesPage(),
    FoldersPage(),
    SettingsPage(),
  ];

  void _onTap(int index) {
    setState(() {
      _currentIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(index: _currentIndex, children: _pages),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _currentIndex,
        onDestinationSelected: _onTap,
        labelBehavior: NavigationDestinationLabelBehavior.alwaysShow,
        surfaceTintColor: Colors.white,
        height: 72,
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.cloud_upload_outlined),
            selectedIcon: Icon(
              Icons.cloud_upload,
              color: AppColors.primaryBlue,
            ),
            label: 'Upload',
          ),
          NavigationDestination(
            icon: Icon(Icons.description_outlined),
            selectedIcon: Icon(Icons.description, color: AppColors.primaryBlue),
            label: 'Notes',
          ),
          NavigationDestination(
            icon: Icon(Icons.folder_outlined),
            selectedIcon: Icon(Icons.folder, color: AppColors.primaryBlue),
            label: 'Folders',
          ),
          NavigationDestination(
            icon: Icon(Icons.settings_outlined),
            selectedIcon: Icon(Icons.settings, color: AppColors.primaryBlue),
            label: 'Settings',
          ),
        ],
      ),
    );
  }
}
